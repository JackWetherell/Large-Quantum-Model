"""Evaluate a ``Stepper`` on one data split and write per-trajectory results plus a summary.

Results file ``results/{run}/{split}.h5`` (shapes: N trajectories, T = n_steps + 1 frames,
S = n_steps one-step positions, M saved trajectories, NX grid points):

t              (T,)          f8  frame times t_k = k * DT (ħ/Hartree)
index          (N,)          i8  row of each trajectory in the split file
seed           (N,)          i8  trajectory seed index (from the split)
family         (N,)          u1  potential family id (constants.FAMILY_IDS)
E, Z, A        (N,)          f8  field, softened-atom charge, perturbation amplitude (from the split)
l1             (N, T)        f8  rollout ∫|n̂ − n| dx
rel_l2         (N, T)        f8  rollout ‖n̂ − n‖₂ / ‖n‖₂
norm_error     (N, T)        f8  rollout ∫n̂ dx − 1
negative_mass  (N, T)        f8  rollout ∫max(−n̂, 0) dx
dipole_pred    (N, T)        f8  rollout ⟨x⟩ = ∫x n̂ dx (Bohr)
dipole_true    (N, T)        f8  exact ⟨x⟩
tf_l1          (N, S)        f8  teacher-forced one-step L1; column k scores the prediction of frame k+1
save_index     (M,)          i8  result rows whose predicted densities are stored (the first M)
n_pred         (M, T, NX)    f8  rollout densities of those rows

Attributes: results_version, model, split, git_hash, config_json.

The exact densities are cast to float64 and each frame is renormalised to ∫n dx = 1 before use,
removing the ~1e-7 normalisation error of the float32 storage, so ``norm_error`` resolves the
float64 conservation of a model (~1e-14) rather than the storage precision. The summary is merged
into ``summary.json`` next to the results file, keyed by the results-file stem (the split name).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from lqm import constants as C
from lqm.eval import metrics as M
from lqm.eval.rollout import rollout, teacher_forced
from lqm.io import load_split, potential
from lqm.model.stepper import Stepper
from lqm.utils import git_hash, set_global_seed

logger = logging.getLogger(__name__)

RESULTS_VERSION: int = 1
SUMMARY_TIMES: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)  # ħ/Hartree
L1_THRESHOLD: float = 0.1

# name -> (shape template, dtype); symbols as in the module docstring.
RESULT_SCHEMA: dict[str, tuple[tuple[str, ...], str]] = {
    "t": (("T",), "f8"),
    "index": (("N",), "i8"),
    "seed": (("N",), "i8"),
    "family": (("N",), "u1"),
    "E": (("N",), "f8"),
    "Z": (("N",), "f8"),
    "A": (("N",), "f8"),
    "l1": (("N", "T"), "f8"),
    "rel_l2": (("N", "T"), "f8"),
    "norm_error": (("N", "T"), "f8"),
    "negative_mass": (("N", "T"), "f8"),
    "dipole_pred": (("N", "T"), "f8"),
    "dipole_true": (("N", "T"), "f8"),
    "tf_l1": (("N", "S"), "f8"),
    "save_index": (("M",), "i8"),
    "n_pred": (("M", "T", "NX"), "f8"),
}


def load_exact(path: str | Path, max_traj: int | None = None) -> dict[str, Any]:
    """Load the first ``max_traj`` trajectories of a split as float64 evaluation targets.

    Returns
    -------
    dict
        ``n`` [N, nt, NX] f8 renormalised per frame to ∫n dx = 1; ``v`` [N, NX] f8 = v0 + E x;
        ``t``, ``seed``, ``family``, ``E``, ``Z``, ``A`` from the file; ``attrs``.
    """
    d = load_split(path, keys=["t", "seed", "family", "E", "Z", "A", "v0", "n"])
    sl = slice(None, max_traj)
    n = d["n"][sl].astype(np.float64)
    n /= n.sum(axis=-1, keepdims=True) * C.DX
    out = {k: d[k][sl] for k in ("seed", "family", "E", "Z", "A")}
    out.update(n=n, v=potential(d["v0"][sl], d["E"][sl]), t=d["t"], attrs=d["attrs"])
    return out


def load_results(path: str | Path) -> dict[str, Any]:
    """Read a results file into ``{name: array}`` plus ``"attrs"`` (with ``"config"`` decoded)."""
    with h5py.File(path, "r") as f:
        out: dict[str, Any] = {k: f[k][...] for k in f}
        attrs = {k: f.attrs[k] for k in f.attrs}
    attrs["config"] = json.loads(attrs["config_json"])
    out["attrs"] = attrs
    return out


def _json_float(x: float) -> float | None:
    """Strict-JSON float: non-finite values become null."""
    x = float(x)
    return x if np.isfinite(x) else None


def _stats(x: np.ndarray, n_boot: int, seed: int) -> dict[str, float | None]:
    ci = M.bootstrap_ci(x, n_boot=n_boot, seed=seed)
    return {
        "mean": _json_float(ci.mean),
        "ci_lo": _json_float(ci.lo),
        "ci_hi": _json_float(ci.hi),
        "median": _json_float(np.median(x)),
        "p90": _json_float(np.quantile(x, 0.9)),
        "max": _json_float(np.max(x)),
    }


def summarise(res: Mapping[str, np.ndarray], n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Headline statistics of one results set, with 95% bootstrap CIs over trajectories.

    Parameters
    ----------
    res : mapping
        Arrays as in a results file (at least ``t``, ``l1``, ``norm_error``, ``negative_mass``,
        ``tf_l1``).
    n_boot, seed : int
        Bootstrap resamples and seed.

    Returns
    -------
    dict
        JSON-serialisable; non-finite numbers are ``null``. ``rollout_l1`` is keyed by the time in
        ħ/Hartree (only times within the horizon). ``time_to_threshold.rmst`` is the restricted
        mean min(t_cross, horizon); ``median`` is null when more than half never cross.
    """
    t, l1 = np.asarray(res["t"]), np.asarray(res["l1"])
    horizon = float(t[-1])
    dt = float(t[1] - t[0])
    out: dict[str, Any] = {
        "n_traj": int(l1.shape[0]),
        "n_steps": int(l1.shape[1] - 1),
        "horizon": horizon,
        "n_nonfinite": int((~np.isfinite(l1)).any(axis=1).sum()),
    }
    out["rollout_l1"] = {
        f"{tq:g}": _stats(l1[:, int(np.argmin(np.abs(t - tq)))], n_boot, seed)
        for tq in SUMMARY_TIMES
        if tq <= horizon + 1e-9
    }
    ttt = M.time_to_threshold(l1, thr=L1_THRESHOLD, dt=dt)
    rmst = M.bootstrap_ci(np.minimum(ttt, horizon), n_boot=n_boot, seed=seed)
    out["time_to_threshold"] = {
        "threshold": L1_THRESHOLD,
        "median": _json_float(np.median(ttt)),
        "p10": _json_float(np.quantile(ttt, 0.1, method="lower")),  # order statistic: no inf - inf
        "frac_censored": float(np.isinf(ttt).mean()),
        "rmst": {"mean": _json_float(rmst.mean), "ci_lo": _json_float(rmst.lo), "ci_hi": _json_float(rmst.hi)},
    }
    out["final_abs_norm_error"] = _stats(np.abs(np.asarray(res["norm_error"])[:, -1]), n_boot, seed)
    out["final_negative_mass"] = _stats(np.asarray(res["negative_mass"])[:, -1], n_boot, seed)
    out["one_step_l1"] = _stats(np.asarray(res["tf_l1"]).mean(axis=1), n_boot, seed)
    return out


def _merge_summary(path: Path, key: str, summary: Mapping[str, Any]) -> None:
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = summary
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def evaluate(
    stepper: Stepper,
    split_path: str | Path,
    out_path: str | Path,
    n_steps: int | None = None,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    max_traj: int | None = None,
    seed: int = 0,
    n_save: int = 32,
    n_boot: int = 1000,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Free rollout + teacher-forced evaluation of ``stepper`` on one split.

    Parameters
    ----------
    stepper : Stepper
        Model to evaluate (already on ``device`` and in eval mode).
    split_path : str or Path
        Data split (``lqm.io`` schema).
    out_path : str or Path
        Results file to write, conventionally ``results/{run}/{split}.h5``.
    n_steps : int, optional
        Rollout length (default nt − 1 = 200).
    batch_size : int
        Trajectories per rollout batch.
    device : str or torch.device, optional
        Device for the rollout (default CUDA if available).
    max_traj : int, optional
        Evaluate only the first ``max_traj`` trajectories.
    seed : int
        Global seed (``lqm.utils.set_global_seed``) and bootstrap seed.
    n_save : int
        Store predicted densities for the first ``n_save`` trajectories.
    n_boot : int
        Bootstrap resamples for the summary CIs.
    meta : mapping, optional
        Extra provenance (e.g. checkpoint path) recorded in the config.

    Returns
    -------
    dict
        The summary (also merged into ``out_path.parent / "summary.json"`` under ``out_path.stem``).
    """
    t_start = time.perf_counter()
    set_global_seed(seed)
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    split_path, out_path = Path(split_path), Path(out_path)

    data = load_exact(split_path, max_traj)
    n_true, v = data["n"], data["v"]
    n_traj, nt, nx = n_true.shape
    n_steps = nt - 1 if n_steps is None else int(n_steps)
    if not 1 <= n_steps <= nt - 1:
        raise ValueError(f"n_steps must be in [1, {nt - 1}], got {n_steps}")
    n_true = n_true[:, : n_steps + 1]
    n_save = min(n_save, n_traj)
    logger.info("evaluate %s on %s: N=%d, n_steps=%d, device=%s", stepper.name, split_path, n_traj, n_steps, device)

    shape = (n_traj, n_steps + 1)
    res: dict[str, np.ndarray] = {k: np.empty(shape) for k in ("l1", "rel_l2", "norm_error", "negative_mass")}
    res["dipole_pred"] = np.empty(shape)
    res["tf_l1"] = np.empty((n_traj, n_steps))
    n_pred = np.empty((n_save, n_steps + 1, nx))
    for start in range(0, n_traj, batch_size):
        sl = slice(start, min(start + batch_size, n_traj))
        true = n_true[sl]
        nb = torch.from_numpy(true).to(device)
        vb = torch.from_numpy(v[sl]).to(device)
        pred = rollout(stepper, nb[:, 0], vb, n_steps).cpu().numpy()
        tf = teacher_forced(stepper, nb[:, :n_steps], vb).cpu().numpy()
        res["l1"][sl] = M.l1(pred, true)
        res["rel_l2"][sl] = M.rel_l2(pred, true)
        res["norm_error"][sl] = M.norm_error(pred)
        res["negative_mass"][sl] = M.negative_mass(pred)
        res["dipole_pred"][sl] = M.dipole(pred)
        res["tf_l1"][sl] = M.l1(tf, true[:, 1:])
        if start < n_save:
            stop = min(sl.stop, n_save)
            n_pred[start:stop] = pred[: stop - start]
        logger.debug("batch %d-%d done", sl.start, sl.stop)

    res.update(
        t=np.asarray(data["t"][: n_steps + 1], dtype=np.float64),
        index=np.arange(n_traj, dtype=np.int64),
        dipole_true=M.dipole(n_true),
        save_index=np.arange(n_save, dtype=np.int64),
        n_pred=n_pred,
        **{k: data[k] for k in ("seed", "family", "E", "Z", "A")},
    )

    config = {
        "model": stepper.name,
        "context": stepper.context,
        "split_path": str(split_path.resolve()),
        "split": str(data["attrs"].get("split", "")),
        "data_git_hash": str(data["attrs"].get("git_hash", "")),
        "n_traj": n_traj,
        "max_traj": max_traj,
        "n_steps": n_steps,
        "dt": float(res["t"][1] - res["t"][0]),
        "batch_size": batch_size,
        "device": str(device),
        "seed": seed,
        "n_boot": n_boot,
        "n_save": n_save,
        "threshold": L1_THRESHOLD,
        "exact_renormalised_f64": True,
        **(dict(meta) if meta else {}),
    }
    code_hash = git_hash()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for name, (_, dtype) in RESULT_SCHEMA.items():
            f.create_dataset(name, data=np.asarray(res[name], dtype=dtype))
        f.attrs.update(
            results_version=RESULTS_VERSION,
            model=stepper.name,
            split=config["split"],
            git_hash=code_hash,
            config_json=json.dumps(config, sort_keys=True),
        )

    summary = summarise(res, n_boot=n_boot, seed=seed)
    config["wall_time_s"] = round(time.perf_counter() - t_start, 3)
    summary.update(git_hash=code_hash, results_file=str(out_path.resolve()), config=config)
    _merge_summary(out_path.parent / "summary.json", out_path.stem, summary)
    logger.info(
        "%s/%s: mean rollout L1 at t=%g: %.3e; mean one-step L1 %.3e; %.1f s",
        stepper.name,
        out_path.stem,
        res["t"][-1],
        res["l1"][:, -1].mean(),
        res["tf_l1"].mean(),
        config["wall_time_s"],
    )
    return summary
