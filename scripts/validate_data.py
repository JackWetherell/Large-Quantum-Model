#!/usr/bin/env python
"""Validate generated LQM-1e dataset files and print (or save) a JSON summary.

Cheap checks run over every trajectory in a file (schema, stored norm, zero current at t = 0,
gap, wall density, spectral weight). Expensive checks regenerate a random subsample of
trajectories in float64 and verify bit-identical reproduction from (base_seed, split, seed),
exact norm and ⟨H1⟩ conservation, the aliasing weight, the discrete continuity equation and the
parity relation n(x, t; v0(−x), −E) = n(−x, t; v0, E), with a conditioning-aware tolerance
max(1e-12, 10·ε·‖H1‖·max n / min(gap0, 1)) so near-degenerate wells are not flagged for
round-off. Distribution checks compare each OOD
split with ``train`` factor by factor (two-sample KS tests), confirming that exactly one factor
moved.

Examples
--------
::

    python scripts/validate_data.py --data data --subsample 32 --json reports/data_checks.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from lqm import constants as C
from lqm.io import check_schema, load_split
from lqm.physics.generate import generate_split
from lqm.physics.propagate import (
    aliasing_weight,
    current,
    density,
    density_rate,
    eigensystem,
    expectation_energy,
    ground_state,
    populations,
    wall_flux,
    wavefunction,
)

logger = logging.getLogger("validate_data")

# Tolerances (see the verification section of the plan).
TOL = {
    "norm_f4": 1e-6,
    "norm_f8": 1e-12,
    "current_t0": 1e-12,
    "wall_flux": 1e-12,
    "energy_rel": 1e-10,
    "continuity_rel": 1e-6,
    "parity": 1e-12,  # floor of the conditioning-aware parity tolerance (see deep_checks)
    "parity_safety": 10.0,
    "alias": 1e-8,
    "min_gap": 1e-3,
    "wall_density": 1e-6,
    "ks_p": 1e-3,
}
CONTINUITY_DELTA = 1e-4  # a.u., for the central difference in time
CHEAP_CHUNK = 1000  # trajectories per chunk in cheap_checks (bounds peak memory)


def _float(x: Any) -> float:
    return float(np.asarray(x).item()) if np.ndim(x) == 0 else float(x)


def _json_default(obj: Any) -> Any:
    """Make numpy scalars and arrays (e.g. HDF5 attributes) JSON-serialisable."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")


def cheap_checks(d: dict[str, Any]) -> dict[str, Any]:
    """Whole-file checks that only use the stored arrays.

    Parameters
    ----------
    d : dict
        Output of ``lqm.io.load_split``.

    Returns
    -------
    dict
        Measured numbers plus ``pass`` booleans.
    """
    # Whole-file reductions are done in chunks of trajectories so that peak memory scales with
    # the chunk, not the split (the unchunked float64 version needed ~14x the f4 size of n).
    norm_f4 = current_t0 = wall_t0 = wall_any = 0.0
    min_density = np.inf
    resid_num = resid_den = 0.0
    for s in range(0, d["n"].shape[0], CHEAP_CHUNK):
        n = d["n"][s : s + CHEAP_CHUNK].astype("f8")
        j = d["j"][s : s + CHEAP_CHUNK].astype("f8")
        norm_f4 = max(norm_f4, np.abs(n.sum(axis=-1) * C.DX - 1.0).max())
        # Time-centred continuity from the stored frames only: O(DT^2) accurate, so this is a
        # consistency smoke test, not a precision test (the precise version is in deep_checks).
        div = np.diff(np.pad(j, [(0, 0), (0, 0), (1, 1)]), axis=-1) / C.DX
        dndt_fd = (n[:, 2:] - n[:, :-2]) / (2.0 * C.DT)
        resid_num = max(resid_num, np.abs(dndt_fd + div[:, 1:-1]).max())
        resid_den = max(resid_den, np.abs(dndt_fd).max())
        min_density = min(min_density, n.min())
        current_t0 = max(current_t0, np.abs(j[:, 0, :]).max())
        wall_t0 = max(wall_t0, n[:, 0, [0, -1]].max())
        wall_any = max(wall_any, n[:, :, [0, -1]].max())
    resid = resid_num / resid_den
    out = {
        "n_trajectories": int(d["n"].shape[0]),
        "families": {name: int((d["family"] == i).sum()) for name, i in C.FAMILY_IDS.items()},
        "norm_err_f4_max": _float(norm_f4),
        "norm_err_stored_max": _float(d["norm_err"].max()),
        "min_density": _float(min_density),
        "current_t0_max": _float(current_t0),
        "gap0_min": _float(d["gap0"].min()),
        "eps0_range": [_float(d["eps0"].min()), _float(d["eps0"].max())],
        "min_v0_range": [_float(d["v0"].min(axis=-1).min()), _float(d["v0"].min(axis=-1).max())],
        "wall_density_t0_max": _float(wall_t0),
        "wall_density_any_t_max": _float(wall_any),
        "spec_weight_missing_max": _float((1.0 - d["spec_w"].sum(axis=-1)).max()),
        "energy_range": [_float(d["energy"].min()), _float(d["energy"].max())],
        "abs_E_range": [_float(np.abs(d["E"]).min()), _float(np.abs(d["E"]).max())],
        "continuity_stored_rel": _float(resid),
    }
    out["pass"] = {
        "norm_f4": out["norm_err_f4_max"] < TOL["norm_f4"],
        "norm_stored": out["norm_err_stored_max"] < TOL["norm_f8"],
        "current_t0": out["current_t0_max"] < TOL["current_t0"],
        "gap0": out["gap0_min"] > TOL["min_gap"],
        "density_non_negative": out["min_density"] >= 0.0,
    }
    return out


def deep_checks(d: dict[str, Any], indices: np.ndarray, split: str, base_seed: int, config: dict) -> dict[str, Any]:
    """Regenerate a subsample in float64 and run the expensive physics checks.

    Parameters
    ----------
    d : dict
        Loaded split.
    indices : ndarray of int
        Positions *within the file* to check.
    split, base_seed, config
        Provenance from the file attributes, used to regenerate trajectories.

    Returns
    -------
    dict
        Measured numbers plus ``pass`` booleans.
    """
    seeds = d["seed"][indices]
    regen, _ = generate_split(split, indices=[int(s) for s in seeds], base_seed=base_seed, config=config)
    bit_identical = bool(np.array_equal(regen["n"], d["n"][indices]) and np.array_equal(regen["j"], d["j"][indices]))

    v0 = d["v0"][indices]
    E = d["E"][indices]
    eps_h0, u0 = ground_state(v0)
    eps, vecs = eigensystem(v0 + E[:, None] * C.X)
    c = populations(vecs, u0)
    u, hu = wavefunction(eps, vecs, c, C.T)
    n = density(u)
    dndt = density_rate(u, hu)
    j = current(dndt)
    e_frames = expectation_energy(u, hu)

    # Continuity at every frame: central difference in time vs the divergence of the stored current.
    frames = np.array([0.0, 1.0, 4.0, 20.0, 40.0])  # a.u.
    up, hup = wavefunction(eps, vecs, c, frames + CONTINUITY_DELTA)
    um, hum = wavefunction(eps, vecs, c, frames - CONTINUITY_DELTA)
    dndt_fd = (density(up) - density(um)) / (2.0 * CONTINUITY_DELTA)
    u_f, hu_f = wavefunction(eps, vecs, c, frames)
    jf = current(density_rate(u_f, hu_f))
    div = np.diff(np.pad(jf, [(0, 0), (0, 0), (1, 1)]), axis=-1) / C.DX
    continuity = np.abs(dndt_fd + div).max() / np.abs(dndt_fd).max()

    v0_p, E_p = np.ascontiguousarray(v0[:, ::-1]), -E
    eps_p, vecs_p = eigensystem(v0_p + E_p[:, None] * C.X)
    _, u0_p = ground_state(v0_p)
    u_p, _ = wavefunction(eps_p, vecs_p, populations(vecs_p, u0_p), C.T)
    # Parity tolerance is conditioning-aware: round-off in the ground-state eigenvector is amplified
    # by ~eps_mach * ||H|| / gap0 (first-order perturbation theory), so near-degenerate wells
    # (e.g. symmetric double wells) legitimately miss a flat tolerance. Per trajectory:
    #   tol_i = max(TOL["parity"], safety * eps_mach * ||H1_i||_2 * max(n_i) / min(gap0_i, 1)).
    # The flat 1e-12 stays as a floor, so well-conditioned trajectories are held to it unchanged.
    parity_i = np.abs(density(u_p) - n[..., ::-1]).max(axis=(-2, -1))
    h_norm = np.abs(eps).max(axis=-1)  # spectral norm of the symmetric H1
    cond = np.finfo(np.float64).eps * h_norm * n.max(axis=(-2, -1)) / np.minimum(d["gap0"][indices], 1.0)
    parity_tol_i = np.maximum(TOL["parity"], TOL["parity_safety"] * cond)
    parity = parity_i.max()

    out = {
        "subsample": [int(s) for s in seeds],
        "bit_identical": bit_identical,
        "norm_err_f8_max": _float(np.abs(n.sum(axis=-1) * C.DX - 1.0).max()),
        "energy_rel_drift_max": _float(
            (
                np.abs(e_frames - e_frames.mean(axis=-1, keepdims=True)).max(axis=-1) / np.abs(e_frames).mean(axis=-1)
            ).max()
        ),
        "energy_vs_stored_max": _float(np.abs(e_frames.mean(axis=-1) - d["energy"][indices]).max()),
        "current_t0_max": _float(np.abs(j[:, 0, :]).max()),
        "wall_flux_max": _float(np.abs(wall_flux(dndt)).max()),
        "continuity_rel": _float(continuity),
        "parity_max": _float(parity),
        "parity_ratio_max": _float((parity_i / parity_tol_i).max()),
        "parity_tol_range": [_float(parity_tol_i.min()), _float(parity_tol_i.max())],
        "parity_n_relaxed": int((parity_tol_i > TOL["parity"]).sum()),
        "alias_max": _float(aliasing_weight(eps, c**2).max()),
        "wall_density_t0_max": _float(density(u0)[:, [0, -1]].max()),
    }
    out["pass"] = {
        "bit_identical": bit_identical,
        "norm_f8": out["norm_err_f8_max"] < TOL["norm_f8"],
        "energy_conserved": out["energy_rel_drift_max"] < TOL["energy_rel"],
        "current_t0": out["current_t0_max"] < TOL["current_t0"],
        "wall_flux": out["wall_flux_max"] < TOL["wall_flux"],
        "continuity": out["continuity_rel"] < TOL["continuity_rel"],
        "parity": bool(np.all(parity_i < parity_tol_i)),
        "alias": out["alias_max"] < TOL["alias"],
    }
    return out


def ood_checks(files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Confirm each OOD split moves exactly one factor relative to ``train``.

    Compares the marginal distributions of Z, A, |E|, min(v0) and eps0 with two-sample KS
    tests. The factor an OOD split is *meant* to change must differ (p below the threshold);
    every other factor must be statistically indistinguishable from training.

    Parameters
    ----------
    files : dict
        ``{split: loaded split}``, must contain ``train``.

    Returns
    -------
    dict
        KS p-values per split and factor, plus ``pass`` booleans.
    """
    if "train" not in files:
        return {"skipped": "train split not loaded"}
    ref = files["train"]
    factors = {
        "Z": lambda d: d["Z"],
        "A": lambda d: d["A"],
        "abs_E": lambda d: np.abs(d["E"]),
        "min_v0": lambda d: d["v0"].min(axis=-1),
        "eps0": lambda d: d["eps0"],
    }
    # Factors each OOD split is expected to change; everything else must match training.
    expected_changed = {"ood_field": {"abs_E"}, "ood_Z": {"Z", "min_v0", "eps0"}, "ood_family": set()}
    out: dict[str, Any] = {}
    for split, d in files.items():
        if not split.startswith("ood_"):
            continue
        p_values = {}
        for name, fn in factors.items():
            a, b = np.asarray(fn(ref), float), np.asarray(fn(d), float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            p_values[name] = _float(ks_2samp(a, b).pvalue) if b.size else float("nan")
        changed = expected_changed[split]
        unchanged_ok = {
            name: bool(p > TOL["ks_p"] or np.isnan(p)) for name, p in p_values.items() if name not in changed
        }
        counts = {name: int((d["family"] == i).sum()) for name, i in C.FAMILY_IDS.items()}
        is_family_split = split == "ood_family"
        # ood_family: no atom_fourier potentials, the three shapes in equal numbers, Z and A NaN.
        n_fam = len(C.FAMILY_IDS) - 1
        family_changed = (
            counts["atom_fourier"] == 0 and max(abs(counts[f] - d["seed"].size / n_fam) for f in C.FAMILIES[1:]) <= 1
        )
        nan_fractions = {k: _float(np.isnan(d[k]).mean()) for k in ("Z", "A")}
        out[split] = {
            "ks_p": p_values,
            "expected_changed": sorted(changed),
            "family_counts": counts,
            "nan_fractions": nan_fractions,
            "pass": {
                "unchanged_factors_match_train": all(unchanged_ok.values()),
                "intended_factor_changed": (
                    family_changed if is_family_split else all(p_values[name] < TOL["ks_p"] for name in changed)
                ),
                "nan_parameters_as_expected": all(
                    f == (1.0 if is_family_split else 0.0) for f in nan_fractions.values()
                ),
            },
            "unchanged_detail": unchanged_ok,
        }
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"), help="directory holding {split}.h5")
    p.add_argument("--splits", nargs="+", default=list(C.SPLITS), choices=list(C.SPLITS))
    p.add_argument("--subsample", type=int, default=16, help="trajectories per split for the expensive checks")
    p.add_argument("--seed", type=int, default=0, help="seed for choosing the subsample")
    p.add_argument("--json", type=Path, default=None, help="write the summary here")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate the requested splits. Returns 0 if every check passed, 1 otherwise."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rng = np.random.default_rng(args.seed)

    summary: dict[str, Any] = {"tolerances": TOL, "splits": {}}
    loaded: dict[str, dict[str, Any]] = {}
    for split in args.splits:
        path = args.data / f"{split}.h5"
        if not path.exists():
            logger.warning("%s missing, skipped", path)
            continue
        problems = check_schema(path)
        d = load_split(path)
        loaded[split] = d
        attrs = d["attrs"]
        n_traj = d["seed"].size
        k = min(args.subsample, n_traj)
        indices = np.sort(rng.choice(n_traj, size=k, replace=False))
        entry = {
            "path": str(path),
            "schema_problems": problems,
            "attrs": {a: attrs[a] for a in ("split", "base_seed", "nt", "nx", "dt", "L", "stencil", "idea_version")},
            "git_hash": attrs["git_hash"],
            "cheap": cheap_checks(d),
            "deep": deep_checks(d, indices, split, int(attrs["base_seed"]), attrs["config"]),
        }
        entry["pass"] = not problems and all(entry["cheap"]["pass"].values()) and all(entry["deep"]["pass"].values())
        summary["splits"][split] = entry
        logger.info(
            "%s: N=%d %s norm_f4=%.2e norm_f8=%.2e energy=%.2e continuity=%.2e parity=%.2e alias=%.2e",
            split,
            n_traj,
            "OK" if entry["pass"] else "FAIL",
            entry["cheap"]["norm_err_f4_max"],
            entry["deep"]["norm_err_f8_max"],
            entry["deep"]["energy_rel_drift_max"],
            entry["deep"]["continuity_rel"],
            entry["deep"]["parity_max"],
            entry["deep"]["alias_max"],
        )
        for name, ok in {**entry["cheap"]["pass"], **entry["deep"]["pass"]}.items():
            if not ok:
                logger.error("%s: check %s FAILED", split, name)

    summary["ood"] = ood_checks(loaded)
    for split, entry in summary["ood"].items():
        if isinstance(entry, dict) and "pass" in entry:
            ok = all(entry["pass"].values())
            logger.info("%s factors: %s %s", split, "OK" if ok else "FAIL", entry["ks_p"])
            if split in summary["splits"]:
                summary["splits"][split]["pass"] = summary["splits"][split]["pass"] and ok

    summary["all_pass"] = bool(summary["splits"]) and all(e["pass"] for e in summary["splits"].values())
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n")
        logger.info("wrote %s", args.json)
    else:
        print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    logger.info("all_pass = %s", summary["all_pass"])
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
