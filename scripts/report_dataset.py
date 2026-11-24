#!/usr/bin/env python
"""Regenerate every figure and number of the LQM-1e dataset report (``reports/01_dataset.md``).

Reads the generated splits in ``data/`` and the validation summary ``reports/data_checks.json``;
writes ``reports/figures/01_*.png`` and ``reports/01_dataset_numbers.json`` (every number quoted in
the report). Atomic units throughout: x in Bohr, t in ħ/Hartree, energies and ω in Hartree, the
field E in Hartree/Bohr, densities in 1/Bohr and currents in 1/(ħ/Hartree).

Deterministic: nothing is sampled at random. Example trajectories are chosen by the documented
criteria in ``select_examples`` / ``select_ood_examples``; subsamples are fixed strides or prefixes;
the only resampling (bootstrap CIs) uses a fixed seed.

Conventions
-----------
* Exact densities are cast to float64 and renormalised per frame to ∫n dx = 1 (as in
  ``lqm.eval.runner.load_exact``) before any error metric is computed.
* "Wall mass" W(t) = ∫_{|x|>8} n dx; W_max = max_t W(t).
* "Displacement" = max_t |⟨x⟩(t) − ⟨x⟩(0)|.
* Persistence one-step L1 at frame k = ∫|n_{k+1} − n_k| dx (teacher forced); persistence rollout L1
  at frame k = ∫|n_k − n_0| dx (free rollout from n_0).
* Dipole lines: with H1 = Σ_a ε_a |a⟩⟨a| and c_a = ⟨a|ψ0⟩ real,
  ⟨x⟩(t) = Σ_a c_a² x_aa + Σ_{a<b} 2 c_a c_b x_ab cos((ε_b − ε_a) t), so every line sits at an H1
  level spacing with amplitude 2 c_a c_b x_ab (Bohr). The H1 eigensystem is recomputed from (v0, E)
  with ``lqm.physics.propagate`` because the stored ``spec_eps``/``spec_w`` carry no matrix elements.

Examples
--------
::

    python scripts/report_dataset.py --data data --out reports
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import matplotlib.lines
import matplotlib.patches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import binned_statistic_2d, ks_2samp

from lqm import constants as C
from lqm.eval import metrics as M
from lqm.eval import plots as P
from lqm.physics.propagate import density, eigensystem, ground_state, populations, wavefunction

logger = logging.getLogger("report_dataset")

# ----------------------------------------------------------------------------------------------
# Analysis parameters (all quoted in the report via the numbers JSON).
WALL_X = 8.0  # Bohr; |x| > WALL_X counts as "near the hard walls" (walls at ±10)
WALL_MASS_THRESHOLD = 0.05  # W_max above this = "reaches the walls"
X_GROUND_OFFSITE = 4.0  # Bohr; |⟨x⟩| of the H1 ground state above this = bound in the tilted region near a wall
L1_THRESHOLD = 0.1  # rollout-error threshold used by the evaluation harness
N_FFT = 8192  # zero-padded FFT length for the dipole spectrum (interpolation only)
LINE_STRIDE = 4  # every LINE_STRIDE-th train trajectory for the H1 eigen-analysis
AMBIG_N = 1000  # first AMBIG_N test trajectories for the look-alike frame search
AMBIG_MIN_SEP = 10  # frames; look-alike pairs must be at least 2 a.u. apart in time
SNAPSHOT_TIMES = (0.0, 10.0, 20.0, 30.0, 40.0)  # a.u.
SMOKE_SUMMARY = Path("runs/smoke_persistence/summary.json")

T_WINDOW = float(C.T[-1])  # 40 a.u.
OMEGA_RES = 2.0 * np.pi / T_WINDOW  # Fourier resolution of a 40 a.u. record, Hartree
WALL_MASK = np.abs(C.X) > WALL_X
SPLITS = C.SPLITS
OOD_SPLITS = ("ood_field", "ood_Z", "ood_family")

# ----------------------------------------------------------------------------------------------
# Colours: one fixed identity per split (train = neutral reference, OOD = categorical slots 1-3,
# validated all-pairs colourblind-safe in lqm.eval.plots' palette); examples are drawn in neutral
# inks, time snapshots in an ordinal one-hue blue ramp.
SPLIT_COLOR = {
    "train": P.INK_MUTED,
    "ood_field": P.CATEGORICAL[0],
    "ood_Z": P.CATEGORICAL[1],
    "ood_family": P.CATEGORICAL[2],
}
SPLIT_LABEL = {
    "train": "train",
    "ood_field": "ood_field (|E| ∈ [0.1, 0.2])",
    "ood_Z": "ood_Z (Z ∈ [0.5,1) ∪ (2,3])",
    "ood_family": "ood_family (new shapes)",
}
SNAPSHOT_COLORS = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
COUNT_CMAP = LinearSegmentedColormap.from_list(  # count maps: the lowest count must stay visible
    "lqm_counts", ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
)
LOG_CMAP = LinearSegmentedColormap.from_list(
    "lqm_logdensity", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def sig(x: Any, n: int = 4) -> Any:
    """Round floats (recursively) to ``n`` significant figures for the numbers JSON."""
    if isinstance(x, dict):
        return {k: sig(v, n) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [sig(v, n) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        if not np.isfinite(x) or x == 0.0:
            return None if not np.isfinite(x) else 0.0
        return float(f"{x:.{n}g}")
    return x


def quantiles(a: np.ndarray, qs: tuple[float, ...] = (0.05, 0.5, 0.95)) -> dict[str, float]:
    """Named quantiles of a 1-D sample, e.g. {'p5': .., 'p50': .., 'p95': ..}."""
    a = np.asarray(a, dtype=float)
    return {f"p{round(100 * q):d}": float(np.quantile(a, q)) for q in qs}


# ----------------------------------------------------------------------------------------------
# Per-trajectory statistics from the stored densities.


def trajectory_stats(n_ds: Any, chunk: int = 1000) -> dict[str, Any]:
    """Dipole, wall mass, persistence/linear-extrapolation errors and spectral peak per trajectory.

    Parameters
    ----------
    n_ds : h5py.Dataset or ndarray
        [N, NT, NX] densities (read chunk by chunk, so the train split never sits in RAM as f64).
    chunk : int
        Trajectories per chunk.

    Returns
    -------
    dict of ndarray
        ``dip`` [N, NT] ⟨x⟩(t); ``wall`` [N, NT] W(t); ``step`` [N, NT-1] ∫|n_{k+1} − n_k|;
        ``lin`` [N, NT-2] ∫|2n_k − n_{k−1} − n_{k+1}| for k = 1..NT-2; ``roll`` [N, NT] ∫|n_k − n_0|;
        ``onset`` [N] ∫|n_2 − n_0| / ∫|n_1 − n_0| (→ 4 if dn/dt(0) = 0, → 2 if not);
        ``omega_peak`` [N] position of the maximum of the Hann-windowed dipole power spectrum;
        ``n_max`` largest stored density value (float).
    """
    n_traj = n_ds.shape[0]
    out = {
        "dip": np.empty((n_traj, C.NT)),
        "wall": np.empty((n_traj, C.NT)),
        "step": np.empty((n_traj, C.NT - 1)),
        "lin": np.empty((n_traj, C.NT - 2)),
        "roll": np.empty((n_traj, C.NT)),
        "onset": np.empty(n_traj),
        "omega_peak": np.empty(n_traj),
        "n_max": 0.0,
    }
    for s in range(0, n_traj, chunk):
        sl = slice(s, min(s + chunk, n_traj))
        n = np.asarray(n_ds[sl], dtype=np.float64)
        out["n_max"] = max(out["n_max"], float(n.max()))
        n /= n.sum(axis=-1, keepdims=True) * C.DX
        out["dip"][sl] = M.dipole(n)
        out["wall"][sl] = n[..., WALL_MASK].sum(axis=-1) * C.DX
        out["step"][sl] = M.l1(n[:, :-1], n[:, 1:])
        out["lin"][sl] = M.l1(2.0 * n[:, 1:-1] - n[:, :-2], n[:, 2:])
        out["roll"][sl] = M.l1(n, n[:, :1])
        out["onset"][sl] = out["roll"][sl, 2] / out["roll"][sl, 1]
        omega, power = M.dipole_spectrum(out["dip"][sl], n_fft=N_FFT)
        out["omega_peak"][sl] = omega[np.argmax(power, axis=-1)]
    return out


def line_analysis(v0: np.ndarray, E: np.ndarray, chunk: int = 500) -> dict[str, np.ndarray]:
    """Exact H1 spectral analysis of the initial state, per trajectory.

    Parameters
    ----------
    v0 : ndarray
        [B, NX] ground-state potentials.
    E : ndarray
        [B] fields.
    chunk : int
        Trajectories per batched diagonalisation.

    Returns
    -------
    dict of ndarray
        ``omega_star`` [B] frequency ε_b − ε_a of the strongest dipole line; ``amp_star`` [B] its
        amplitude |2 c_a c_b x_ab| (Bohr); ``frac_star`` [B] its share Σ-normalised of amp² over all
        lines; ``w_ground`` [B] population c_0² of the H1 ground state; ``x_ground`` [B] ⟨x⟩ of
        the H1 ground state (Bohr); ``ground_most_populated`` [B] bool; ``norm_h1`` [B] spectral
        norm max|ε_a| of H1 (Hartree); ``eps_a``, ``eps_b`` [B] the two levels of the strongest line.
    """
    n_traj = v0.shape[0]
    keys = ("omega_star", "amp_star", "frac_star", "w_ground", "x_ground", "norm_h1", "eps_a", "eps_b")
    out = {k: np.empty(n_traj) for k in keys}
    out["ground_most_populated"] = np.empty(n_traj, dtype=bool)
    for s in range(0, n_traj, chunk):
        sl = slice(s, min(s + chunk, n_traj))
        _, u0 = ground_state(v0[sl])
        eps, vecs = eigensystem(v0[sl] + E[sl, None] * C.X)
        c = populations(vecs, u0)
        xab = np.swapaxes(vecs, -1, -2) @ (C.X[:, None] * vecs)
        amp = np.triu(2.0 * c[:, :, None] * c[:, None, :] * xab, k=1)
        flat = np.abs(amp).reshape(amp.shape[0], -1)
        best = np.argmax(flat, axis=-1)
        a, b = np.unravel_index(best, (C.NX, C.NX))
        rows = np.arange(amp.shape[0])
        out["omega_star"][sl] = eps[rows, b] - eps[rows, a]
        out["eps_a"][sl], out["eps_b"][sl] = eps[rows, a], eps[rows, b]
        out["amp_star"][sl] = flat[rows, best]
        out["frac_star"][sl] = flat[rows, best] ** 2 / (flat**2).sum(axis=-1)
        out["w_ground"][sl] = c[:, 0] ** 2
        out["x_ground"][sl] = xab[:, 0, 0]
        out["ground_most_populated"][sl] = np.argmax(c**2, axis=-1) == 0
        out["norm_h1"][sl] = np.abs(eps).max(axis=-1)
    return out


def dipole_lines(v0: np.ndarray, E: float) -> dict[str, np.ndarray]:
    """All dipole lines of one trajectory: frequencies, signed amplitudes, static part and spectrum.

    Returns
    -------
    dict
        ``omega`` [L], ``amp`` [L] (signed, Bohr) for a < b; ``static`` (Bohr); ``eps`` [NX] H1
        levels; ``w`` [NX] populations; ``u0`` [NX] initial state (ℓ2).
    """
    _, u0 = ground_state(v0[None])
    eps, vecs = eigensystem((v0 + E * C.X)[None])
    c = populations(vecs, u0)[0]
    eps, vecs = eps[0], vecs[0]
    xab = vecs.T @ (C.X[:, None] * vecs)
    iu = np.triu_indices(C.NX, 1)
    amp = 2.0 * c[iu[0]] * c[iu[1]] * xab[iu]
    return {
        "omega": eps[iu[1]] - eps[iu[0]],
        "amp": amp,
        "static": float((c**2 * np.diag(xab)).sum()),
        "eps": eps,
        "w": c**2,
        "u0": u0[0],
    }


# ----------------------------------------------------------------------------------------------
# Example selection (documented criteria).

EXAMPLE_TARGETS = {
    "A": ("typical", 1.5, 0.05),
    "B": ("weak field", 1.5, 0.01),
    "C": ("strong field, deep well", 1.9, 0.095),
}


def nearest(Z: np.ndarray, E: np.ndarray, z_target: float, e_target: float, mask: np.ndarray | None = None) -> int:
    """Index minimising the distance in (Z, |E|) with |E| scaled by its training range 0.1."""
    dist = np.hypot(Z - z_target, (np.abs(E) - e_target) / 0.1)
    if mask is not None:
        dist = np.where(mask, dist, np.inf)
    return int(np.argmin(dist))


def select_examples(Z: np.ndarray, E: np.ndarray, wall_max: np.ndarray) -> dict[str, dict[str, Any]]:
    """Four test-set examples: A-C nearest to target (Z, |E|); D = the largest wall mass W_max."""
    ex = {}
    for key, (label, z_t, e_t) in EXAMPLE_TARGETS.items():
        ex[key] = {"label": label, "index": nearest(Z, E, z_t, e_t), "criterion": f"nearest to Z={z_t}, |E|={e_t}"}
    ex["D"] = {
        "label": "weak binding + strong field (max wall contact)",
        "index": int(np.argmax(wall_max)),
        "criterion": "argmax over test of W_max",
    }
    return ex


# ----------------------------------------------------------------------------------------------
# Plot helpers.


def finish_axes(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def snapshot_indices() -> list[int]:
    return [int(round(t / C.DT)) for t in SNAPSHOT_TIMES]


def draw_potential(
    ax_v: plt.Axes,
    ax_n: plt.Axes,
    v0: np.ndarray,
    E: float,
    n: np.ndarray,
    eps0: float,
    title: str,
    band: tuple[np.ndarray, np.ndarray] | None = None,
    ylim: tuple[float, float] = (-2.3, 1.3),
) -> None:
    """Top: v0, v0 + E·x and ε0 (optional training 5-95% envelope). Bottom: n(x, t) snapshots."""
    if band is not None:
        ax_v.fill_between(C.X, band[0], band[1], color=P.GRID, linewidth=0, label="train v0, 5–95%")
    ax_v.plot(C.X, v0, color=P.INK_MUTED, linestyle="--", linewidth=1.2, label="v0 (t < 0)")
    ax_v.plot(C.X, v0 + E * C.X, color=P.INK, linewidth=1.5, label="v0 + E·x (t > 0)")
    ax_v.axhline(eps0, color=P.INK_SECONDARY, linewidth=0.8, linestyle=":", label="ε0 of H0")
    ax_v.set_ylim(*ylim)
    ax_v.set_xlim(-C.L, C.L)
    ax_v.set_title(title, fontsize=9)
    for k, color in zip(snapshot_indices(), SNAPSHOT_COLORS, strict=True):
        ax_n.plot(C.X, n[k], color=color, linewidth=1.3, label=f"t = {C.T[k]:g}")
    ax_n.set_xlim(-C.L, C.L)
    ax_n.set_ylim(bottom=0.0)
    finish_axes(ax_n, P.X_LABEL)


def ecdf(ax: plt.Axes, values: np.ndarray, split: str, **kw: Any) -> None:
    v = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, v.size + 1) / v.size
    style = {"color": SPLIT_COLOR[split], "linewidth": 2.4 if split == "train" else 1.6, "label": SPLIT_LABEL[split]}
    style.update(kw)
    ax.step(v, y, where="post", **style)


# ----------------------------------------------------------------------------------------------
# Figures. Each returns the numbers it quotes.


def fig_potentials(test: dict, ex: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharex=True, layout="constrained")
    for col, (key, e) in enumerate(ex.items()):
        i = e["index"]
        title = f"{key}: {e['label']}\nZ = {test['Z'][i]:.2f}, E = {test['E'][i]:+.3f}, A = {test['A'][i]:.2f}"
        draw_potential(axes[0, col], axes[1, col], test["v0"][i], test["E"][i], test["n"][i], test["eps0"][i], title)
    axes[0, 0].set_ylabel("potential (Ha)")
    axes[1, 0].set_ylabel("density n (1/a.u.)")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    axes[1, 0].legend(loc="upper left", fontsize=8)
    P.save_figure(fig, fig_dir / "01_example_potentials.png")


def fig_heatmaps(test: dict, ex: dict, stats: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(15, 11), sharex=True, sharey=True, layout="constrained")
    extent_n = (C.X[0] - C.DX / 2, C.X[-1] + C.DX / 2, C.T[0] - C.DT / 2, C.T[-1] + C.DT / 2)
    x_face = 0.5 * (C.X[:-1] + C.X[1:])
    extent_j = (x_face[0] - C.DX / 2, x_face[-1] + C.DX / 2, extent_n[2], extent_n[3])
    common = {"origin": "lower", "aspect": "auto", "interpolation": "nearest"}
    vmax_n = max(float(test["n"][e["index"]].max()) for e in ex.values())
    for col, (key, e) in enumerate(ex.items()):
        i = e["index"]
        n, j = test["n"][i], test["j"][i]
        im_n = axes[0, col].imshow(n, cmap=P.DENSITY_CMAP, vmin=0.0, vmax=vmax_n, extent=extent_n, **common)
        axes[0, col].plot(stats["dip"][i], C.T, color=P.INK, linewidth=0.9, label="⟨x⟩(t)")
        axes[0, col].set_title(f"{key}: {e['label']}\nZ = {test['Z'][i]:.2f}, E = {test['E'][i]:+.3f}", fontsize=9)
        log_n = np.log10(np.maximum(n, 1e-12))
        im_l = axes[1, col].imshow(log_n, cmap=LOG_CMAP, vmin=-6.0, vmax=0.0, extent=extent_n, **common)
        jmax = float(np.abs(j).max())
        im_j = axes[2, col].imshow(j, cmap=P.DIFF_CMAP, vmin=-jmax, vmax=jmax, extent=extent_j, **common)
        cb = fig.colorbar(im_j, ax=axes[2, col], orientation="horizontal", shrink=0.85, pad=0.02)
        tick = float(f"{0.9 * jmax:.1g}")
        cb.set_ticks([-tick, 0.0, tick])
        cb.set_label("current j (a.u.), own scale per panel")
        for r in range(3):
            axes[r, col].grid(False)
        for xw in (-WALL_X, WALL_X):
            axes[1, col].axvline(xw, color=P.INK_SECONDARY, linewidth=0.6, linestyle=":")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    for r in range(3):
        axes[r, 0].set_ylabel(P.T_LABEL)
    for c in range(4):
        axes[2, c].set_xlabel(P.X_LABEL)
    fig.colorbar(im_n, ax=axes[0, :], shrink=0.9, label="density n (1/a.u.)")
    fig.colorbar(im_l, ax=axes[1, :], shrink=0.9, label="log10 n (n in 1/a.u.)", extend="min")
    P.save_figure(fig, fig_dir / "01_heatmaps.png")


def fig_parameters(par: dict[str, dict], fig_dir: Path) -> None:
    panels = [
        ("Z", "Z (a.u.)", False),
        ("E", "field E (Ha/a.u.)", False),
        ("A", "perturbation amplitude A (Ha)", False),
        ("min_v0", "min v0 (Ha)", False),
        ("eps0", "ε0 of H0 (Ha)", False),
        ("gap0", "gap ε1 − ε0 of H0 (Ha), log", True),
        ("energy", "⟨H1⟩ (Ha)", False),
        ("depletion", "1 − w_g: population outside H1 ground state, log", True),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), layout="constrained")
    for p_i, (ax, (key, label, logx)) in enumerate(zip(axes.flat, panels, strict=True)):
        vals = {s: np.asarray(par[s][key], float) for s in ("train", *OOD_SPLITS)}
        finite = np.concatenate([v[np.isfinite(v)] for v in vals.values()])
        if logx:
            finite = finite[finite > 0]
            lo, hi = max(finite.min(), 1e-7), finite.max()
            bins = np.logspace(np.log10(lo), np.log10(hi), 50)
            ax.set_xscale("log")
        else:
            bins = np.linspace(finite.min(), finite.max(), 50)
        for s, v in vals.items():
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            weights = np.full(v.size, 1.0 / v.size)  # fraction per bin: comparable across splits and log bins
            if s == "train":
                ax.hist(v, bins=bins, weights=weights, color=SPLIT_COLOR[s], alpha=0.45, linewidth=0)
            else:
                ax.hist(v, bins=bins, weights=weights, histtype="step", color=SPLIT_COLOR[s], linewidth=1.5)
        finish_axes(ax, label, "fraction of split per bin" if p_i % 4 == 0 else None)
    handles = [
        matplotlib.patches.Patch(color=SPLIT_COLOR["train"], alpha=0.45, label=SPLIT_LABEL["train"]),
        *(matplotlib.lines.Line2D([], [], color=SPLIT_COLOR[s], label=SPLIT_LABEL[s]) for s in OOD_SPLITS),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    P.save_figure(fig, fig_dir / "01_parameter_distributions.png")


def fig_dipole_examples(test: dict, ex: dict, stats: dict, fig_dir: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(3, 4, figsize=(15, 8.5), layout="constrained")
    out = {}
    omega_max = 1.2
    for col, (key, e) in enumerate(ex.items()):
        i = e["index"]
        d = stats["dip"][i]
        lines = dipole_lines(test["v0"][i], float(test["E"][i]))
        recon = lines["static"] + (lines["amp"][None, :] * np.cos(lines["omega"][None, :] * C.T[:, None])).sum(-1)
        omega, power = M.dipole_spectrum(d, n_fft=N_FFT)
        order = np.argsort(-np.abs(lines["amp"]))
        strong = order[np.abs(lines["amp"][order]) > 0.02 * np.abs(lines["amp"][order[0]])]
        w_star = float(lines["omega"][order[0]])
        # stored spectrum: are the strongest line's two levels among the 16 stored levels?
        spec = test["spec_eps"][i]
        iu = np.triu_indices(C.NX, 1)
        ea, eb = lines["eps"][iu[0][order[0]]], lines["eps"][iu[1][order[0]]]
        in_spec = bool(np.isclose(spec, ea, atol=1e-10).any() and np.isclose(spec, eb, atol=1e-10).any())
        out[key] = {
            "index": i,
            "omega_star": w_star,
            "period_star": 2 * np.pi / w_star,
            "omega_peak": float(stats["omega_peak"][i]),
            "amp_star": float(np.abs(lines["amp"][order[0]])),
            "n_lines_above_2pct": int(strong.size),
            "strongest_line_levels_in_spec": in_spec,
            "reconstruction_max_err": float(np.abs(recon - d).max()),
            "displacement": float(np.abs(d - d[0]).max()),
        }
        ax = axes[0, col]
        ax.plot(C.T, d, color=P.INK, linewidth=1.4, label="⟨x⟩ from n")
        ax.plot(C.T, recon, color=P.INK_MUTED, linewidth=1.0, linestyle="--", label="Σ of H1 lines")
        ax.set_title(f"{key}: {e['label']}\nZ = {test['Z'][i]:.2f}, E = {test['E'][i]:+.3f}", fontsize=9)
        finish_axes(ax, P.T_LABEL, "⟨x⟩ (a.u.)" if col == 0 else None)
        ax = axes[1, col]
        sel = omega <= omega_max
        ax.plot(omega[sel], power[sel], color=P.INK, linewidth=1.3)
        ax.axvline(w_star, color=P.INK_MUTED, linewidth=0.9, linestyle="--", label="strongest line ω*")
        ax.set_xlim(0, omega_max)
        ax.set_ylim(bottom=0)
        finish_axes(ax, None, "|d̃(ω)|² (a.u.)" if col == 0 else None)
        ax = axes[2, col]
        ws, amps = lines["omega"][strong], np.abs(lines["amp"][strong])
        keep = ws <= omega_max
        ax.vlines(ws[keep], 0, amps[keep], color=P.INK, linewidth=1.6)
        ax.plot(ws[keep], amps[keep], "o", color=P.INK, markersize=3.5)
        ax.axvspan(0, OMEGA_RES, color=P.GRID, linewidth=0, label="ω < 2π/40 (unresolved)")
        ax.set_xlim(0, omega_max)
        ax.set_ylim(bottom=0)
        finish_axes(ax, "ω = ε_b − ε_a (Ha)", "|2 c_a c_b x_ab| (a.u.)" if col == 0 else None)
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[1, 0].legend(loc="upper right", fontsize=8)
    axes[2, 0].legend(loc="upper right", fontsize=8)
    P.save_figure(fig, fig_dir / "01_dipole_spectra.png")
    return out


def fig_frequency(par_train: dict, lines: dict, stats_train: dict, idx: np.ndarray, fig_dir: Path) -> dict[str, Any]:
    w_star = lines["omega_star"]
    w_peak = stats_train["omega_peak"][idx]
    period = 2 * np.pi / w_star
    resolved = w_star >= OMEGA_RES
    err = np.abs(w_peak - w_star)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    ax = axes[0]
    hb = ax.hexbin(w_star, w_peak, gridsize=60, bins="log", cmap=COUNT_CMAP, mincnt=1, extent=(0, 0.8, 0, 0.8))
    ax.plot([0, 0.8], [0, 0.8], color=P.INK, linewidth=0.8, label="ω_peak = ω*")
    ax.axvspan(0, OMEGA_RES, color=P.GRID, alpha=0.6, linewidth=0, label="ω* < 2π/40")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(hb, ax=ax, label="trajectories per cell (log)")
    finish_axes(ax, "strongest H1 dipole line ω* (Ha)", "peak of windowed dipole spectrum (Ha)")
    ax = axes[1]
    frames = period / C.DT
    bins = np.logspace(np.log10(frames.min()), np.log10(frames.max()), 60)
    ax.hist(frames, bins=bins, color=SPLIT_COLOR["train"], alpha=0.6, linewidth=0)
    ax.set_xscale("log")
    ax.axvspan(50, 100, color=P.CATEGORICAL[0], alpha=0.12, linewidth=0, label="10–20 a.u. (50–100 frames)")
    ax.axvline(T_WINDOW / C.DT, color=P.INK, linewidth=0.9, linestyle="--", label="trajectory length (200 frames)")
    ax.legend(loc="upper right", fontsize=8)
    finish_axes(ax, "dominant period 2π/ω* (frames of Δt = 0.2 a.u.)", "trajectories")
    ax = axes[2]
    Z, absE = par_train["Z"][idx], np.abs(par_train["E"][idx])
    med, xe, ye, _ = binned_statistic_2d(Z, absE, w_star, statistic="median", bins=[10, 10], range=[[1, 2], [0, 0.1]])
    pm = ax.pcolormesh(xe, ye, med.T, cmap=P.DENSITY_CMAP, shading="flat")
    fig.colorbar(pm, ax=ax, label="median ω* (Ha)")
    ax.grid(False)
    finish_axes(ax, "Z (a.u.)", "|E| (Ha/a.u.)")
    P.save_figure(fig, fig_dir / "01_frequency_check.png")
    return {
        "n_traj": int(idx.size),
        "stride": LINE_STRIDE,
        "omega_res": OMEGA_RES,
        "omega_star": quantiles(w_star, (0.05, 0.25, 0.5, 0.75, 0.95)),
        "period_star_au": quantiles(period),
        "period_star_frames": quantiles(frames),
        "frac_period_10_20_au": float(((period >= 10) & (period <= 20)).mean()),
        "frac_unresolved_omega_lt_res": float((~resolved).mean()),
        "peak_err_median": float(np.median(err)),
        "frac_peak_within_0p05": float((err < 0.05).mean()),
        "frac_peak_within_0p05_resolved": float((err[resolved] < 0.05).mean()),
        "frac_star_power_median": float(np.median(lines["frac_star"])),
        "frac_star_power_p10": float(np.quantile(lines["frac_star"], 0.1)),
        "frac_ground_most_populated": float(lines["ground_most_populated"].mean()),
        "frac_h1_ground_offsite": float((np.abs(lines["x_ground"]) > X_GROUND_OFFSITE).mean()),
        "frac_wall_with_offsite_ground": float(
            (np.abs(lines["x_ground"]) > X_GROUND_OFFSITE)[stats_train["wall"][idx].max(1) > WALL_MASS_THRESHOLD].mean()
        ),
        "offsite_agrees_with_not_most_populated": float(
            ((np.abs(lines["x_ground"]) > X_GROUND_OFFSITE) == ~lines["ground_most_populated"]).mean()
        ),
        "frac_strongest_line_levels_in_spec": float(
            (
                np.isclose(par_train["spec_eps"][idx], lines["eps_a"][:, None], atol=1e-10).any(1)
                & np.isclose(par_train["spec_eps"][idx], lines["eps_b"][:, None], atol=1e-10).any(1)
            ).mean()
        ),
        "w_ground": quantiles(lines["w_ground"]),
        "frac_not_ground_and_wall": float(
            ((~lines["ground_most_populated"]) & (stats_train["wall"][idx].max(1) > WALL_MASS_THRESHOLD)).sum()
            / max(1, (~lines["ground_most_populated"]).sum())
        ),
    }


def fig_dynamics_stats(par: dict, stats: dict, fig_dir: Path) -> dict[str, Any]:
    disp = np.abs(stats["dip"] - stats["dip"][:, :1]).max(1)
    wmax = stats["wall"].max(1)
    step_med = np.median(stats["step"], axis=1)
    Z, absE = par["Z"], np.abs(par["E"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout="constrained")
    axes = axes.ravel()
    ax = axes[0]
    bins = np.logspace(np.log10(disp.min()), np.log10(disp.max()), 60)
    ax.hist(disp, bins=bins, color=SPLIT_COLOR["train"], alpha=0.6, linewidth=0)
    ax.axvline(np.median(disp), color=P.INK, linewidth=0.9, linestyle="--", label=f"median {np.median(disp):.2f}")
    ax.set_xscale("log")
    ax.legend(loc="upper left", fontsize=8)
    finish_axes(ax, "max_t |⟨x⟩(t) − ⟨x⟩(0)| (a.u.)", "trajectories")
    rng = [[1, 2], [0, 0.1]]
    frac, xe, ye, _ = binned_statistic_2d(
        Z, absE, (wmax > WALL_MASS_THRESHOLD).astype(float), statistic="mean", bins=[10, 10], range=rng
    )
    ax = axes[1]
    pm = ax.pcolormesh(xe, ye, frac.T, cmap=P.DENSITY_CMAP, vmin=0, vmax=1, shading="flat")
    fig.colorbar(pm, ax=ax, label=f"fraction with W_max > {WALL_MASS_THRESHOLD}")
    ax.set_title("reaches the walls: max_t ∫_{|x|>8} n dx > 0.05", fontsize=9)
    ax.grid(False)
    finish_axes(ax, "Z (a.u.)", "|E| (Ha/a.u.)")
    med, xe, ye, _ = binned_statistic_2d(Z, absE, step_med, statistic="median", bins=[10, 10], range=rng)
    ax = axes[2]
    pm = ax.pcolormesh(xe, ye, med.T, cmap=P.DENSITY_CMAP, shading="flat")
    fig.colorbar(pm, ax=ax, label="median ∫|n_{k+1} − n_k| dx")
    ax.set_title("per-frame change (= persistence one-step L1)", fontsize=9)
    ax.grid(False)
    finish_axes(ax, "Z (a.u.)", "|E| (Ha/a.u.)")
    ax = axes[3]
    allsteps = stats["step"].ravel()
    bins = np.logspace(-7, np.log10(allsteps.max()), 70)
    w_all = np.full(allsteps.size, 1.0 / allsteps.size)
    w_first = np.full(stats["step"].shape[0], 1.0 / stats["step"].shape[0])
    ax.hist(allsteps, bins=bins, weights=w_all, color=SPLIT_COLOR["train"], alpha=0.6, linewidth=0, label="all frames")
    ax.hist(stats["step"][:, 0], bins=bins, weights=w_first, histtype="step", color=P.INK, linewidth=1.4)
    ax.plot([], [], color=P.INK, label="first step (t = 0 → 0.2) only")
    ax.set_xscale("log")
    ax.legend(loc="upper left", fontsize=8)
    finish_axes(ax, "∫|n_{k+1} − n_k| dx", "fraction per bin")
    P.save_figure(fig, fig_dir / "01_dynamics_statistics.png")
    corner = (Z < 1.2) & (absE > 0.08)
    return {
        "corner_def": "Z < 1.2 and |E| > 0.08",
        "corner_frac_of_split": float(corner.mean()),
        "frac_wall_in_corner": float((wmax[corner] > WALL_MASS_THRESHOLD).mean()),
        "frac_wall_outside_corner": float((wmax[~corner] > WALL_MASS_THRESHOLD).mean()),
        "share_of_wall_trajs_with_absE_gt_0p06": float(
            ((wmax > WALL_MASS_THRESHOLD) & (absE > 0.06)).sum() / (wmax > WALL_MASS_THRESHOLD).sum()
        ),
        "min_absE_with_wall": float(absE[wmax > WALL_MASS_THRESHOLD].min()),
        "max_Z_with_wall": float(Z[wmax > WALL_MASS_THRESHOLD].max()),
        "step_median_bin_range": [float(np.nanmin(med)), float(np.nanmax(med))],
    }


def lookalike_search(n: np.ndarray, j: np.ndarray, min_sep: int = AMBIG_MIN_SEP) -> dict[str, np.ndarray]:
    """For each trajectory, the pair of frames (≥ min_sep apart) with the most similar densities.

    Returns
    -------
    dict of ndarray
        ``k``, ``kp`` frame indices (k < kp); ``d`` = ∫|n_k − n_kp|; ``step_med`` = the trajectory's
        median one-step change; ``jcorr`` = ⟨j_k, j_kp⟩ / (‖j_k‖‖j_kp‖); ``d_next`` = ∫|n_{k+1} − n_{kp+1}|.
    """
    kk = np.arange(C.NT)
    far = np.abs(kk[:, None] - kk[None, :]) >= min_sep
    far &= (kk[:, None] < C.NT - 1) & (kk[None, :] < C.NT - 1)  # both must have a next frame
    out = {k: np.empty(n.shape[0]) for k in ("d", "step_med", "jcorr", "d_next")}
    out["k"] = np.empty(n.shape[0], dtype=int)
    out["kp"] = np.empty(n.shape[0], dtype=int)
    for b in range(n.shape[0]):
        nb = n[b]
        dist = np.abs(nb[:, None, :] - nb[None, :, :]).sum(-1) * C.DX
        dist = np.where(far, dist, np.inf)
        k, kp = sorted(np.unravel_index(int(np.argmin(dist)), dist.shape))
        jk, jkp = j[b, k], j[b, kp]
        out["k"][b], out["kp"][b] = k, kp
        out["d"][b] = dist[k, kp]
        out["step_med"][b] = np.median(np.abs(np.diff(nb, axis=0)).sum(-1) * C.DX)
        norm = np.sqrt((jk @ jk) * (jkp @ jkp))
        out["jcorr"][b] = (jk @ jkp) / norm if norm > 0 else 0.0  # j(t=0) = 0 exactly: no direction
        out["d_next"][b] = np.abs(nb[k + 1] - nb[kp + 1]).sum() * C.DX
    return out


def fig_lookalike(test: dict, stats: dict, fig_dir: Path) -> dict[str, Any]:
    n = np.asarray(test["n"][:AMBIG_N], dtype=np.float64)
    n /= n.sum(-1, keepdims=True) * C.DX
    j = np.asarray(test["j"][:AMBIG_N], dtype=np.float64)
    res = lookalike_search(n, j)
    ratio = res["d"] / res["step_med"]
    lookalike = (ratio < 1.0) & (res["jcorr"] < -0.5)
    # Example: among look-alike pairs closer than half a typical step with nearly opposite currents,
    # the one with the largest current magnitude (most visible).
    jmag = np.array([np.abs(j[b, res["k"][b]]).sum() * C.DX for b in range(n.shape[0])])
    cand = (ratio < 0.5) & (res["jcorr"] < -0.9)
    b = int(np.argmax(np.where(cand, jmag, -np.inf))) if cand.any() else int(np.argmin(res["jcorr"]))
    k, kp = int(res["k"][b]), int(res["kp"][b])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), layout="constrained")
    ax = axes[0]
    ax.plot(C.T, stats["dip"][b], color=P.INK, linewidth=1.3)
    for kk, style, lab in ((k, "o", f"frame k = {k}"), (kp, "s", f"frame k' = {kp}")):
        ax.plot(C.T[kk], stats["dip"][b, kk], style, color=P.INK, markersize=7, markerfacecolor=P.GRID, label=lab)
    ax.legend(loc="best", fontsize=8)
    finish_axes(ax, P.T_LABEL, "⟨x⟩ (a.u.)")
    ax.set_title(f"test trajectory {b}: Z = {test['Z'][b]:.2f}, E = {test['E'][b]:+.3f}", fontsize=9)
    ax = axes[1]
    ax.plot(C.X, n[b, k], color=P.INK, linewidth=1.6, label=f"n at t = {C.T[k]:.1f}")
    ax.plot(C.X, n[b, kp], color=P.INK_MUTED, linewidth=1.6, linestyle="--", label=f"n at t = {C.T[kp]:.1f}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-C.L, C.L)
    ax.set_ylim(bottom=0)
    finish_axes(ax, P.X_LABEL, "density n (1/a.u.)")
    ax.set_title(f"∫|n_k − n_k'| dx = {res['d'][b]:.1e}  (median step {res['step_med'][b]:.1e})", fontsize=9)
    ax = axes[2]
    xf = 0.5 * (C.X[:-1] + C.X[1:])
    ax.axhline(0, color=P.AXIS, linewidth=0.8)
    ax.plot(xf, j[b, k], color=P.INK, linewidth=1.6, label=f"j at t = {C.T[k]:.1f}")
    ax.plot(xf, j[b, kp], color=P.INK_MUTED, linewidth=1.6, linestyle="--", label=f"j at t = {C.T[kp]:.1f}")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(-C.L, C.L)
    finish_axes(ax, P.X_LABEL, "current j (a.u.)")
    ax.set_title(f"current correlation {res['jcorr'][b]:+.2f}", fontsize=9)
    P.save_figure(fig, fig_dir / "01_lookalike_frames.png")
    return {
        "n_traj": int(n.shape[0]),
        "min_sep_frames": AMBIG_MIN_SEP,
        "frac_pair_closer_than_median_step": float((ratio < 1.0).mean()),
        "frac_lookalike_opposite_current": float(lookalike.mean()),
        "ratio_d_next_over_d_median": float(np.median(res["d_next"] / res["d"])),
        "ratio_d_next_over_d_median_lookalike": float(np.median((res["d_next"] / res["d"])[lookalike])),
        "example": {
            "index": b,
            "Z": float(test["Z"][b]),
            "E": float(test["E"][b]),
            "k": k,
            "kp": kp,
            "t_k": float(C.T[k]),
            "t_kp": float(C.T[kp]),
            "d": float(res["d"][b]),
            "step_med": float(res["step_med"][b]),
            "jcorr": float(res["jcorr"][b]),
            "d_next": float(res["d_next"][b]),
        },
    }


def fig_persistence(stats: dict, fig_dir: Path) -> dict[str, Any]:
    step, lin, roll = stats["step"], stats["lin"], stats["roll"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
    ax = axes[0]
    t_pred = C.T[1:]
    for arr, t, name in ((step, t_pred, "persistence"), (lin, C.T[2:], "linear")):
        style = P.REFERENCE_STYLES[name]
        lo, hi = np.quantile(arr, [0.05, 0.95], axis=0)
        if name == "persistence":
            ax.fill_between(t, lo, hi, color=style["color"], alpha=0.2, linewidth=0, label="persistence 5–95%")
            ax.plot(t, arr.mean(0), label="persistence n̂ = n_k (mean)", **style)
        else:
            ax.plot(t, lo, color=style["color"], linewidth=0.7, label="linear 5% and 95%")
            ax.plot(t, hi, color=style["color"], linewidth=0.7)
            ax.plot(t, arr.mean(0), label="linear extrapolation n̂ = 2n_k − n_{k−1} (mean)", **style)
    ax.set_yscale("log")
    ax.set_ylim(2e-5, 1.0)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    finish_axes(ax, "time of predicted frame t_{k+1} (a.u.)", "one-step L1 = ∫|n̂_{k+1} − n_{k+1}| dx")
    ax = axes[1]
    lo, hi = np.quantile(roll, [0.05, 0.95], axis=0)
    style = P.REFERENCE_STYLES["persistence"]
    ax.fill_between(C.T, lo, hi, color=style["color"], alpha=0.18, linewidth=0, label="5–95% of trajectories")
    ax.plot(C.T, roll.mean(0), color=P.INK, linewidth=1.5, label="mean")
    ax.plot(C.T, np.median(roll, 0), color=P.INK_SECONDARY, linewidth=1.2, linestyle="--", label="median")
    ax.axhline(L1_THRESHOLD, color=P.INK_MUTED, linewidth=0.8, linestyle="-.", label=f"L1 = {L1_THRESHOLD}")
    ax.set_xlim(0, T_WINDOW)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=8)
    finish_axes(ax, P.T_LABEL, "persistence rollout L1 = ∫|n_0 − n(t)| dx")
    P.save_figure(fig, fig_dir / "01_persistence_baseline.png")

    ttt = M.time_to_threshold(roll, thr=L1_THRESHOLD)
    one_step_traj = step.mean(1)
    ci = M.bootstrap_ci(one_step_traj, seed=0)
    out: dict[str, Any] = {
        "n_traj": int(step.shape[0]),
        "one_step_all_frames": {**quantiles(step.ravel(), (0.1, 0.5, 0.9)), "mean": float(step.mean())},
        "one_step_traj_mean": {"mean": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi},
        "one_step_first_frame_median": float(np.median(step[:, 0])),
        "one_step_late_median_k_ge_10": float(np.median(step[:, 10:])),
        "linear_one_step_all_frames": {**quantiles(lin.ravel(), (0.1, 0.5, 0.9)), "mean": float(lin.mean())},
        "linear_over_persistence_median_ratio": float(np.median(lin.ravel()) / np.median(step[:, 1:].ravel())),
        # Rough bar: if one-step errors added up linearly, staying below L1_THRESHOLD for all 200 steps
        # would need a one-step L1 below L1_THRESHOLD / 200.
        "linear_accumulation_budget": L1_THRESHOLD / (C.NT - 1),
        "one_step_mean_peak": {"value": float(step.mean(0).max()), "t": float(t_pred[np.argmax(step.mean(0))])},
        "rollout_mean_peak": {"value": float(roll.mean(0).max()), "t": float(C.T[np.argmax(roll.mean(0))])},
        "rollout": {},
        "time_to_threshold": {
            "median": float(np.median(ttt)),
            "p10": float(np.quantile(ttt, 0.1)),
            "frac_never": float(np.isinf(ttt).mean()),
        },
    }
    for tq in (1.0, 5.0, 10.0, 20.0, 40.0):
        k = int(round(tq / C.DT))
        out["rollout"][f"{tq:g}"] = {
            "mean": float(roll[:, k].mean()),
            **quantiles(roll[:, k], (0.05, 0.5, 0.9, 0.95)),
            "max": float(roll[:, k].max()),
        }
    # Cross-check against the evaluation harness (runs/smoke_persistence: first 256 test trajectories).
    if SMOKE_SUMMARY.exists():
        smoke = json.loads(SMOKE_SUMMARY.read_text())["test"]
        n_s = int(smoke["n_traj"])
        out["smoke_crosscheck"] = {
            "n_traj": n_s,
            "harness_one_step_mean": smoke["one_step_l1"]["mean"],
            "numpy_one_step_mean": float(step[:n_s].mean()),
            "harness_rollout_t40_mean": smoke["rollout_l1"]["40"]["mean"],
            "numpy_rollout_t40_mean": float(roll[:n_s, -1].mean()),
        }
    return out


def select_ood_examples(ood: dict[str, dict]) -> list[tuple[str, int, str]]:
    """One example per OOD factor/family, chosen by the documented targets."""
    f, z, fam = ood["ood_field"], ood["ood_Z"], ood["ood_family"]
    cols = [
        ("ood_field", nearest(f["Z"], f["E"], 1.5, 0.15), "ood_field: strong field"),
        ("ood_Z", nearest(z["Z"], z["E"], 0.75, 0.05), "ood_Z: weak binding"),
        ("ood_Z", nearest(z["Z"], z["E"], 2.5, 0.05), "ood_Z: deep well"),
    ]
    for fid, name in ((1, "double_well"), (2, "harmonic"), (3, "pure_fourier")):
        m = fam["family"] == fid
        i = int(np.argmin(np.where(m, np.abs(np.abs(fam["E"]) - 0.05), np.inf)))
        cols.append(("ood_family", i, f"ood_family: {name}"))
    return cols


def fig_ood_potentials(ood: dict, band: tuple[np.ndarray, np.ndarray], fig_dir: Path) -> list[dict[str, Any]]:
    cols = select_ood_examples(ood)
    fig, axes = plt.subplots(2, len(cols), figsize=(18, 6.0), sharex=True, layout="constrained")
    out = []
    for c, (split, i, title) in enumerate(cols):
        d = ood[split]
        zt = "" if np.isnan(d["Z"][i]) else f"Z = {d['Z'][i]:.2f}, "
        full = f"{title}\n{zt}E = {d['E'][i]:+.3f}, gap0 = {d['gap0'][i]:.3f}"
        draw_potential(axes[0, c], axes[1, c], d["v0"][i], d["E"][i], d["n"][i], d["eps0"][i], full, band, (-3.2, 1.6))
        out.append({"split": split, "index": i, "title": title, "E": float(d["E"][i]), "gap0": float(d["gap0"][i])})
    axes[0, 0].set_ylabel("potential (Ha)")
    axes[1, 0].set_ylabel("density n (1/a.u.)")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    axes[1, 0].legend(loc="upper left", fontsize=7)
    P.save_figure(fig, fig_dir / "01_ood_potentials.png")
    return out


def kohn_check(fam: dict, stats: dict) -> dict[str, Any]:
    """Harmonic OOD potentials: ⟨x⟩(t) must equal ⟨x⟩(0) − (E/ω²)(1 − cos ωt) (generalised Kohn theorem)."""
    m = np.where(fam["family"] == C.FAMILY_IDS["harmonic"])[0]
    curvature = np.polyfit(C.X, fam["v0"][m].T, 2)[0] * 2.0
    omega = np.sqrt(curvature)
    dip = stats["dip"][m]
    pred = dip[:, :1] - (fam["E"][m] / omega**2)[:, None] * (1.0 - np.cos(omega[:, None] * C.T[None, :]))
    return {
        "indices": m,
        "omega": omega,
        "pred": pred,
        "max_err": float(np.abs(pred - dip).max()),
        "max_amplitude": float(np.abs(dip - dip[:, :1]).max()),
        "omega_range": [float(omega.min()), float(omega.max())],
        "max_abs_gap0_minus_omega": float(np.abs(fam["gap0"][m] - omega).max()),
        "v0_wall_max": float(fam["v0"][m][:, [0, -1]].max()),
    }


def fig_ood_dynamics(stats: dict[str, dict], kohn: dict, ood_lines: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), layout="constrained")
    specs = [
        (axes[0, 0], lambda s: np.abs(stats[s]["dip"] - stats[s]["dip"][:, :1]).max(1), "displacement (a.u.)", True),
        (
            axes[0, 1],
            lambda s: np.maximum(stats[s]["wall"].max(1), 1e-10),
            "W_max = max_t ∫_{|x|>8} n dx (floor 1e-10)",
            True,
        ),
        (axes[0, 2], lambda s: np.median(stats[s]["step"], 1), "median one-step persistence L1", True),
        (axes[1, 0], lambda s: stats[s]["roll"][:, -1], "persistence rollout L1 at t = 40", False),
        (axes[1, 1], lambda s: ood_lines[s]["omega_star"], "strongest dipole line ω* (Ha)", False),
    ]
    for ax, fn, label, logx in specs:
        for s in ("train", *OOD_SPLITS):
            ecdf(ax, fn(s), s)
        if logx:
            ax.set_xscale("log")
        finish_axes(ax, label, "fraction of trajectories ≤ x")
        ax.set_ylim(0, 1.01)
    axes[0, 1].axvline(WALL_MASS_THRESHOLD, color=P.INK, linewidth=0.8, linestyle=":")
    axes[1, 1].axvspan(0, OMEGA_RES, color=P.GRID, linewidth=0)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    ax = axes[1, 2]
    m = kohn["indices"]
    om = kohn["omega"]  # three examples spanning the frequency range: lowest, median and highest ω
    order = [int(np.argmin(om)), int(np.argsort(om)[om.size // 2]), int(np.argmax(om))]
    for r, o in enumerate(order):
        d = stats["ood_family"]["dip"][m[o]]
        ax.plot(C.T[::4], d[::4] - d[0], "o", color=P.INK, markersize=3.2, label="⟨x⟩ from data" if r == 0 else None)
        ax.plot(
            C.T,
            kohn["pred"][o] - d[0],
            color=P.CATEGORICAL[2],
            linewidth=1.3,
            label="−(E/ω²)(1 − cos ωt)" if r == 0 else None,
        )
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.45 * np.ptp(ax.get_ylim()))
    ax.legend(loc="upper center", fontsize=8, ncol=2)
    ax.set_title(f"harmonic family: max |error| = {kohn['max_err']:.1e} a.u. over all 333", fontsize=9)
    finish_axes(ax, P.T_LABEL, "⟨x⟩(t) − ⟨x⟩(0) (a.u.)")
    P.save_figure(fig, fig_dir / "01_ood_dynamics.png")


# ----------------------------------------------------------------------------------------------
# Summaries.


def split_summary(par: dict, st: dict, lines: dict | None, path: Path) -> dict[str, Any]:
    disp = np.abs(st["dip"] - st["dip"][:, :1]).max(1)
    wmax = st["wall"].max(1)
    out: dict[str, Any] = {
        "N": int(par["E"].size),
        "file_GB": path.stat().st_size / 1e9,
        "Z_range": [float(np.nanmin(par["Z"])), float(np.nanmax(par["Z"]))] if np.isfinite(par["Z"]).any() else None,
        "absE_range": [float(np.abs(par["E"]).min()), float(np.abs(par["E"]).max())],
        "A_range": [float(np.nanmin(par["A"])), float(np.nanmax(par["A"]))] if np.isfinite(par["A"]).any() else None,
        "min_v0_range": [float(par["min_v0"].min()), float(par["min_v0"].max())],
        "eps0_range": [float(par["eps0"].min()), float(par["eps0"].max())],
        "gap0": {"min": float(par["gap0"].min()), **quantiles(par["gap0"])},
        "energy_range": [float(par["energy"].min()), float(par["energy"].max())],
        "dipole0_abs_median": float(np.median(np.abs(st["dip"][:, 0]))),
        "displacement": {**quantiles(disp, (0.5, 0.9)), "max": float(disp.max())},
        "frac_wall_gt_thr": float((wmax > WALL_MASS_THRESHOLD).mean()),
        "frac_wall_gt_0p01": float((wmax > 0.01).mean()),
        "wall_max_max": float(wmax.max()),
        "wall_t0_max": float(st["wall"][:, 0].max()),
        "n_max": st["n_max"],
        "one_step_median": float(np.median(st["step"])),
        "rollout_t40": {"mean": float(st["roll"][:, -1].mean()), **quantiles(st["roll"][:, -1], (0.5, 0.9))},
        "onset_ratio_median": float(np.median(st["onset"])),
        "sudden_energy_check_max": float(np.abs(par["energy"] - (par["eps0"] + par["E"] * st["dip"][:, 0])).max()),
    }
    if lines is not None:
        out["frac_ground_not_most_populated"] = float((~lines["ground_most_populated"]).mean())
        out["frac_h1_ground_offsite"] = float((np.abs(lines["x_ground"]) > X_GROUND_OFFSITE).mean())
        out["omega_star"] = quantiles(lines["omega_star"])
        out["frac_unresolved_omega"] = float((lines["omega_star"] < OMEGA_RES).mean())
        out["norm_h1_max"] = float(lines["norm_h1"].max())
    return out


def parity_diagnostic(fam: dict, checks: dict) -> dict[str, Any]:
    """Recompute the validator's parity residual on its ood_family subsample, grouped by family.

    Same construction as ``scripts/validate_data.py``: densities of (v0(−x), −E) vs n(−x), float64.
    """
    seeds = np.asarray(checks["splits"]["ood_family"]["deep"]["subsample"])
    pos = np.searchsorted(fam["seed"], seeds)
    v0, E = fam["v0"][pos], fam["E"][pos]
    _, u0 = ground_state(v0)
    eps, vecs = eigensystem(v0 + E[:, None] * C.X)
    n = density(wavefunction(eps, vecs, populations(vecs, u0), C.T)[0])
    v0p, Ep = np.ascontiguousarray(v0[:, ::-1]), -E
    _, u0p = ground_state(v0p)
    epsp, vecsp = eigensystem(v0p + Ep[:, None] * C.X)
    n_p = density(wavefunction(epsp, vecsp, populations(vecsp, u0p), C.T)[0])
    resid = np.abs(n_p - n[..., ::-1]).max(axis=(1, 2))
    norm_h1 = np.abs(eps).max(-1)
    out: dict[str, Any] = {"n_subsample": int(pos.size), "tolerance": checks["tolerances"]["parity"]}
    for name in C.FAMILIES[1:]:
        m = fam["family"][pos] == C.FAMILY_IDS[name]
        if m.any():
            out[name] = {
                "n": int(m.sum()),
                "parity_max": float(resid[m].max()),
                "parity_median": float(np.median(resid[m])),
                "norm_h1_max": float(norm_h1[m].max()),
                "norm_h1_median": float(np.median(norm_h1[m])),
                "n_over_tol": int((resid[m] > checks["tolerances"]["parity"]).sum()),
            }
    gap0 = fam["gap0"][pos]
    out["corr_log_resid_log_normH"] = float(np.corrcoef(np.log(resid), np.log(norm_h1))[0, 1])
    out["corr_log_resid_log_gap0"] = float(np.corrcoef(np.log(resid), np.log(gap0))[0, 1])
    w = int(np.argmax(resid))
    out["worst"] = {
        "seed": int(fam["seed"][pos[w]]),
        "family": C.FAMILIES[int(fam["family"][pos[w]])],
        "parity": float(resid[w]),
        "gap0": float(gap0[w]),
        "norm_h1": float(norm_h1[w]),
        "eps_mach_normH_over_gap0": float(np.finfo(float).eps * norm_h1[w] / gap0[w]),
        "is_min_gap0_of_split": bool(np.isclose(gap0[w], fam["gap0"].min())),
    }
    out["second_largest_parity"] = float(np.sort(resid)[-2])
    return out


def validation_summary(checks: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"tolerances": checks["tolerances"], "all_pass": checks["all_pass"], "splits": {}}
    for s, e in checks["splits"].items():
        cheap, deep = e["cheap"], e["deep"]
        failed = [k for k, ok in {**cheap["pass"], **deep["pass"]}.items() if not ok]
        out["splits"][s] = {
            "pass": e["pass"],
            "failed": failed,
            "n_deep": len(deep["subsample"]),
            "norm_f4_max": cheap["norm_err_f4_max"],
            "norm_f8_max": deep["norm_err_f8_max"],
            "energy_rel_drift_max": deep["energy_rel_drift_max"],
            "continuity_rel": deep["continuity_rel"],
            "parity_max": deep["parity_max"],
            "alias_max": deep["alias_max"],
            "wall_flux_max": deep["wall_flux_max"],
            "current_t0_max": cheap["current_t0_max"],
            "bit_identical": deep["bit_identical"],
            "min_density": cheap["min_density"],
            "gap0_min": cheap["gap0_min"],
            "spec_weight_missing_max": cheap["spec_weight_missing_max"],
            "git_hash": e["git_hash"],
        }
    out["ood_ks_p"] = {s: checks["ood"][s]["ks_p"] for s in OOD_SPLITS}
    out["ood_pass"] = {s: checks["ood"][s]["pass"] for s in OOD_SPLITS}
    return out


# ----------------------------------------------------------------------------------------------


def load_params(path: Path) -> dict[str, np.ndarray]:
    keys = ("seed", "family", "Z", "E", "A", "v0", "eps0", "gap0", "energy", "spec_eps", "spec_w")
    with h5py.File(path, "r") as f:
        d = {k: f[k][...] for k in keys}
    d["min_v0"] = d["v0"].min(axis=-1)
    return d


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"), help="directory with {split}.h5")
    p.add_argument("--out", type=Path, default=Path("reports"), help="report directory")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t_start = time.perf_counter()
    P.apply_style()
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    checks = json.loads((args.out / "data_checks.json").read_text())

    par: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    for s in SPLITS:
        path = args.data / f"{s}.h5"
        par[s] = load_params(path)
        with h5py.File(path, "r") as f:
            stats[s] = trajectory_stats(f["n"])
        logger.info("%s: trajectory statistics done (%.0f s)", s, time.perf_counter() - t_start)

    line_idx = np.arange(0, par["train"]["E"].size, LINE_STRIDE)
    lines = {"train": line_analysis(par["train"]["v0"][line_idx], par["train"]["E"][line_idx])}
    for s in ("test", *OOD_SPLITS):
        lines[s] = line_analysis(par[s]["v0"], par[s]["E"])
    logger.info("H1 line analysis done (%.0f s)", time.perf_counter() - t_start)
    par["train"]["depletion"] = np.full(par["train"]["E"].size, np.nan)  # only the strided subsample
    par["train"]["depletion"][line_idx] = 1.0 - lines["train"]["w_ground"]
    for s in OOD_SPLITS:
        par[s]["depletion"] = 1.0 - lines[s]["w_ground"]

    with h5py.File(args.data / "test.h5", "r") as f:
        test = {**par["test"], "n": f["n"][...].astype(np.float64), "j": f["j"][...].astype(np.float64)}
    ood = {}
    for s in OOD_SPLITS:
        with h5py.File(args.data / f"{s}.h5", "r") as f:
            ood[s] = {**par[s], "n": f["n"][...].astype(np.float64)}

    num: dict[str, Any] = {
        "meta": {
            "script": "scripts/report_dataset.py",
            "data_git_hash": checks["splits"]["train"]["git_hash"],
            "idea_version": checks["splits"]["train"]["attrs"]["idea_version"],
            "wall_x": WALL_X,
            "wall_mass_threshold": WALL_MASS_THRESHOLD,
            "x_ground_offsite": X_GROUND_OFFSITE,
            "l1_threshold": L1_THRESHOLD,
            "n_fft": N_FFT,
            "line_stride_train": LINE_STRIDE,
            "omega_res": OMEGA_RES,
            "nyquist_omega": float(np.pi / C.DT),
            "bf16_unit_roundoff": 2.0**-8,
            "f32_unit_roundoff": 2.0**-24,
            "n_input_target_pairs_train": int(par["train"]["E"].size * (C.NT - 1)),
        },
        "grid": {"nx": C.NX, "L": C.L, "dx": C.DX, "dt": C.DT, "nt": C.NT, "t_max": T_WINDOW, "stencil": C.STENCIL},
    }
    num["splits"] = {s: split_summary(par[s], stats[s], lines.get(s), args.data / f"{s}.h5") for s in SPLITS}
    num["splits"]["train"]["frac_top_spec_not_lowest_spec"] = float(
        (np.argmax(par["train"]["spec_w"], 1) != np.argmin(par["train"]["spec_eps"], 1)).mean()
    )
    num["total_GB"] = sum(num["splits"][s]["file_GB"] for s in SPLITS)
    num["ks_train_vs"] = {
        s: {
            k: float(ks_2samp(par["train"][k], par[s][k]).pvalue)
            for k in ("Z", "E", "A", "eps0", "gap0", "min_v0", "energy")
        }
        for s in ("val", "test")
    }
    num["validation"] = validation_summary(checks)
    num["density_range"] = {
        "min_over_splits": min(num["validation"]["splits"][s]["min_density"] for s in SPLITS),
        "max_over_splits": max(stats[s]["n_max"] for s in SPLITS),
    }

    # Physics checks beyond data_checks.json.
    tr_idx = np.arange(0, 3000, 300)
    _, u0 = ground_state(test["v0"][tr_idx])
    eps, vecs = eigensystem(test["v0"][tr_idx] + test["E"][tr_idx, None] * C.X)
    c = populations(vecs, u0)
    n_plus = density(wavefunction(eps, vecs, c, C.T)[0])
    n_minus = density(wavefunction(eps, vecs, c, -C.T)[0])
    num["physics_checks"] = {
        "time_reversal_n_minus_t_max": float(np.abs(n_plus - n_minus).max()),
        "time_reversal_n_traj": int(tr_idx.size),
        "initial_state_is_real": bool(np.isrealobj(u0)),
        "onset_ratio_train": quantiles(stats["train"]["onset"]),
    }

    ex = select_examples(test["Z"], test["E"], stats["test"]["wall"].max(1))
    fig_potentials(test, ex, fig_dir)
    fig_heatmaps(test, ex, stats["test"], fig_dir)
    num["examples"] = {
        k: {
            **e,
            "Z": float(test["Z"][e["index"]]),
            "E": float(test["E"][e["index"]]),
            "A": float(test["A"][e["index"]]),
            "gap0": float(test["gap0"][e["index"]]),
            "wall_max": float(stats["test"]["wall"][e["index"]].max()),
            "j_absmax": float(np.abs(test["j"][e["index"]]).max()),
            "w_ground": float(lines["test"]["w_ground"][e["index"]]),
            "x_ground": float(lines["test"]["x_ground"][e["index"]]),
        }
        for k, e in ex.items()
    }
    spectra = fig_dipole_examples(test, ex, stats["test"], fig_dir)
    for k in ex:
        num["examples"][k].update(spectra[k])
    num["physics_checks"]["dipole_line_reconstruction_max_err"] = max(
        spectra[k]["reconstruction_max_err"] for k in spectra
    )
    fig_parameters(par, fig_dir)
    num["frequency_train"] = fig_frequency(par["train"], lines["train"], stats["train"], line_idx, fig_dir)
    num["dynamics_train"] = fig_dynamics_stats(par["train"], stats["train"], fig_dir)
    num["lookalike_test"] = fig_lookalike(test, stats["test"], fig_dir)
    num["persistence_test"] = fig_persistence(stats["test"], fig_dir)
    logger.info("in-distribution figures done (%.0f s)", time.perf_counter() - t_start)

    band = tuple(np.quantile(par["train"]["v0"], [0.05, 0.95], axis=0))
    num["ood_examples"] = fig_ood_potentials(ood, band, fig_dir)
    kohn = kohn_check(par["ood_family"], stats["ood_family"])
    fig_ood_dynamics(stats, kohn, lines, fig_dir)
    num["kohn_harmonic"] = {k: v for k, v in kohn.items() if k not in ("indices", "omega", "pred")}
    fam = par["ood_family"]
    num["ood_family_by_family"] = {}
    for name in C.FAMILIES[1:]:
        m = fam["family"] == C.FAMILY_IDS[name]
        st = stats["ood_family"]
        num["ood_family_by_family"][name] = {
            "n": int(m.sum()),
            "gap0": {"min": float(fam["gap0"][m].min()), **quantiles(fam["gap0"][m])},
            "frac_gap0_lt_0p05": float((fam["gap0"][m] < 0.05).mean()),
            "v0_wall_max": float(fam["v0"][m][:, [0, -1]].max()),
            "displacement_median": float(np.median(np.abs(st["dip"][m] - st["dip"][m, :1]).max(1))),
            "frac_wall_gt_thr": float((st["wall"][m].max(1) > WALL_MASS_THRESHOLD).mean()),
            "rollout_t40_median": float(np.median(st["roll"][m, -1])),
            "norm_h1_max": float(lines["ood_family"]["norm_h1"][m].max()),
        }
    zz = par["ood_Z"]["Z"]
    num["ood_Z_by_side"] = {
        side: {
            "n": int(m.sum()),
            "displacement_median": float(
                np.median(np.abs(stats["ood_Z"]["dip"][m] - stats["ood_Z"]["dip"][m, :1]).max(1))
            ),
            "frac_wall_gt_thr": float((stats["ood_Z"]["wall"][m].max(1) > WALL_MASS_THRESHOLD).mean()),
            "rollout_t40_median": float(np.median(stats["ood_Z"]["roll"][m, -1])),
            "eps0_max": float(par["ood_Z"]["eps0"][m].max()),
        }
        for side, m in (("Z_lt_1", zz < 1.0), ("Z_gt_2", zz > 2.0))
    }
    num["parity_diagnostic"] = parity_diagnostic(par["ood_family"], checks)
    num["train_parity_norm_h1_max"] = num["splits"]["train"]["norm_h1_max"]

    out_json = args.out / "01_dataset_numbers.json"
    out_json.write_text(json.dumps(sig(num), indent=2, sort_keys=False, allow_nan=False) + "\n")
    logger.info("wrote %s; total %.0f s", out_json, time.perf_counter() - t_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
