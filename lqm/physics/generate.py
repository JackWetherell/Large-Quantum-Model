"""Generation of the exact LQM-1e dataset splits (atomic units).

``generate_split`` samples potentials with :mod:`lqm.physics.potentials`, propagates them
exactly with :mod:`lqm.physics.propagate` and returns arrays that satisfy ``lqm.io.SCHEMA``
together with the attributes required by ``lqm.io.REQUIRED_ATTRS``.

All physics is float64; only the stored ``n`` and ``j`` are cast to float32. Work is done in
chunks of trajectories so that batched LAPACK/BLAS calls stay cache-friendly and peak memory
stays bounded; ``numpy.linalg.eigh`` is bit-identical for a stacked batch and for a single
matrix, so a trajectory regenerated on its own reproduces the stored data exactly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Any

import numpy as np

from lqm import constants as C
from lqm.physics import potentials
from lqm.physics.propagate import simulate
from lqm.utils import git_hash, trajectory_rng

logger = logging.getLogger(__name__)

_TRAIN_SPEC: dict[str, Any] = {
    "family": "atom_fourier",
    "z_range": [1.0, 2.0],
    "amp_range": [0.0, 0.2],
    "n_modes": 8,
    "field_range": [-0.1, 0.1],
    "accept": {"min_gap": 1e-3},
}


def default_config() -> dict[str, Any]:
    """Return the generation configuration: grid, protocol, counts and split distributions.

    Each split specification is self-contained, so that the single factor an OOD split changes
    is explicit in the stored ``config_json``: ``ood_field`` replaces ``field_range`` by
    ``field_abs_range``, ``ood_Z`` replaces ``z_range`` by ``z_intervals`` and ``ood_family``
    replaces the ``atom_fourier`` shape by one of three others (the ``atom_fourier`` keys stay,
    because a reference draw from that distribution fixes each OOD potential's depth, see
    :func:`lqm.physics.potentials.match_binding`).

    Returns
    -------
    dict
        JSON-serialisable configuration.
    """
    return {
        "schema_version": C.SCHEMA_VERSION,
        "base_seed": 1234,
        "counts": {"train": 24000, "val": 3000, "test": 3000, "ood_field": 1000, "ood_Z": 1000, "ood_family": 1000},
        "grid": {"nx": C.NX, "L": C.L, "dx": C.DX, "dt": C.DT, "nt": C.NT, "stencil": C.STENCIL},
        "protocol": (
            "1 electron, 1D, hard walls. Ground state of H0 = H(v0) for t < 0; at t = 0 a uniform field E is "
            "switched on suddenly and held constant, H1 = H(v0 + E x). Exact propagation by diagonalising H1."
        ),
        "n_spec": C.N_SPEC,
        "max_alias_weight": 1e-8,
        "max_attempts": 64,
        "splits": {
            "train": dict(_TRAIN_SPEC),
            "val": dict(_TRAIN_SPEC),
            "test": dict(_TRAIN_SPEC),
            "ood_field": {**_TRAIN_SPEC, "field_range": None, "field_abs_range": [0.1, 0.2]},
            "ood_Z": {**_TRAIN_SPEC, "z_range": None, "z_intervals": [[0.5, 1.0], [2.0, 3.0]]},
            "ood_family": {
                **_TRAIN_SPEC,
                "family": "ood_family",
                "families": list(potentials.FAMILY_SHAPES),
                "harmonic": {"center_range": [-0.5, 0.5]},
                "double_well": {
                    "separation_range": [1.5, 3.5],
                    "width_range": [0.5, 1.0],
                    "asymmetry_range": [-0.3, 0.3],
                },
                "pure_fourier": {"n_modes": 8},
                "accept": {"min_gap": 1e-3, "max_wall_density": 1e-6},
            },
        },
    }


def _split_spec(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Return the split specification with ``None`` placeholders dropped."""
    spec = config["splits"][split]
    return {k: v for k, v in spec.items() if v is not None}


def generate_split(
    split: str,
    n: int | None = None,
    base_seed: int = 1234,
    *,
    indices: Sequence[int] | None = None,
    chunk_size: int = 128,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate one split (or an arbitrary subset of its trajectories).

    Parameters
    ----------
    split : str
        One of ``constants.SPLITS``.
    n : int, optional
        Number of trajectories, i.e. indices ``0..n-1``. Ignored if ``indices`` is given.
    base_seed : int
        Base seed; trajectory *i* uses ``trajectory_rng(base_seed, split, i)``.
    indices : sequence of int, optional
        Explicit trajectory indices, for regenerating single trajectories. The stored ``seed``
        dataset holds these indices.
    chunk_size : int
        Trajectories propagated per batched call (peak memory ≈ 4 × chunk_size × nt × NX × 16 B).
    config : mapping, optional
        Configuration as returned by :func:`default_config`.

    Returns
    -------
    arrays : dict of ndarray
        Every dataset of ``lqm.io.SCHEMA`` with its exact shape and dtype.
    attrs : dict
        Every attribute of ``lqm.io.REQUIRED_ATTRS``, including the full ``config_json``.

    Raises
    ------
    ValueError
        If any trajectory exceeds ``config["max_alias_weight"]``: that would mean the frame
        spacing ``DT`` undersamples a populated beat frequency.
    """
    config = dict(default_config() if config is None else config)
    spec = _split_spec(config, split)
    if indices is None:
        if n is None:
            raise ValueError("give either n or indices")
        indices = range(int(n))
    idx = np.asarray(list(indices), dtype=np.int64)
    n_traj = idx.size
    n_spec = int(config["n_spec"])
    max_attempts = int(config["max_attempts"])

    t_start = time.perf_counter()
    v0 = np.empty((n_traj, C.NX), dtype=float)
    fields = np.empty(n_traj, dtype=float)
    charges = np.empty(n_traj, dtype=float)
    amplitudes = np.empty(n_traj, dtype=float)
    families = np.empty(n_traj, dtype=np.uint8)
    attempts = np.empty(n_traj, dtype=np.int64)
    for k, i in enumerate(idx):
        p = potentials.sample(spec, int(i), trajectory_rng(base_seed, split, int(i)), max_attempts)
        v0[k], fields[k], charges[k], amplitudes[k] = p.v0, p.E, p.Z, p.A
        families[k] = C.FAMILY_IDS[p.family]
        attempts[k] = p.attempts
    t_sample = time.perf_counter() - t_start
    logger.info(
        "%s: sampled %d potentials in %.1f s (mean %.2f draws, max %d)",
        split,
        n_traj,
        t_sample,
        attempts.mean() if n_traj else np.nan,
        attempts.max() if n_traj else 0,
    )

    arrays: dict[str, np.ndarray] = {
        "x": np.array(C.X, dtype="f8"),
        "t": np.array(C.T, dtype="f8"),
        "seed": idx,
        "family": families,
        "Z": charges,
        "E": fields,
        "A": amplitudes,
        "v0": v0,
        "n": np.empty((n_traj, C.NT, C.NX), dtype="f4"),
        "j": np.empty((n_traj, C.NT, C.NFACE), dtype="f4"),
        "energy": np.empty(n_traj, dtype="f8"),
        "eps0": np.empty(n_traj, dtype="f8"),
        "gap0": np.empty(n_traj, dtype="f8"),
        "spec_eps": np.empty((n_traj, n_spec), dtype="f8"),
        "spec_w": np.empty((n_traj, n_spec), dtype="f8"),
        "norm_err": np.empty(n_traj, dtype="f8"),
    }
    alias = np.empty(n_traj, dtype=float)
    drift = np.empty(n_traj, dtype=float)
    wall = np.empty(n_traj, dtype=float)

    t_prop = time.perf_counter()
    for start in range(0, n_traj, chunk_size):
        stop = min(start + chunk_size, n_traj)
        out = simulate(v0[start:stop], fields[start:stop], C.T, n_spec)
        arrays["n"][start:stop] = out["n"].astype("f4")
        arrays["j"][start:stop] = out["j"].astype("f4")
        for key in ("energy", "eps0", "gap0", "spec_eps", "spec_w", "norm_err"):
            arrays[key][start:stop] = out[key]
        alias[start:stop] = out["alias"]
        drift[start:stop] = out["energy_drift"]
        wall[start:stop] = np.abs(out["wall_flux"]).max(axis=-1)
        logger.info("%s: propagated %d/%d", split, stop, n_traj)
    t_prop = time.perf_counter() - t_prop

    if n_traj:
        if alias.max() > config["max_alias_weight"]:
            bad = int(np.argmax(alias))
            raise ValueError(
                f"{split}: aliased population {alias.max():.3g} > {config['max_alias_weight']:.3g} "
                f"at index {idx[bad]}: DT undersamples the dynamics"
            )
        logger.info(
            "%s: norm_err %.2e, alias %.2e, energy drift %.2e, wall flux %.2e, "
            "gap0 min %.3f, generated in %.1f s (%.1f s sampling, %.1f s propagation)",
            split,
            arrays["norm_err"].max(),
            alias.max(),
            drift.max(),
            wall.max(),
            arrays["gap0"].min(),
            t_sample + t_prop,
            t_sample,
            t_prop,
        )

    attrs = {
        "schema_version": C.SCHEMA_VERSION,
        "split": split,
        "base_seed": int(base_seed),
        "nt": C.NT,
        "nx": C.NX,
        "dt": C.DT,
        "L": C.L,
        "stencil": C.STENCIL,
        "idea_version": version("iDEA-latest"),
        "git_hash": git_hash(),
        "config_json": config,
        "family_names": C.FAMILIES,
    }
    return arrays, attrs


def generate_trajectory(
    split: str,
    index: int,
    base_seed: int = 1234,
    config: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Regenerate a single trajectory, bit-identical to the stored one.

    Parameters
    ----------
    split : str
        Split name.
    index : int
        Trajectory index (the value stored in the ``seed`` dataset).
    base_seed : int
        Base seed of the dataset (stored as the ``base_seed`` attribute).
    config : mapping, optional
        Configuration (stored as ``config_json``).

    Returns
    -------
    dict of ndarray
        The schema arrays with a leading axis of length 1.
    """
    arrays, _ = generate_split(split, indices=[int(index)], base_seed=base_seed, config=config)
    return arrays
