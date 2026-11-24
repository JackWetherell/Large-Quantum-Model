"""HDF5 dataset schema and I/O for LQM-1e (frozen interface).

One file per split, ``data/{split}.h5``. Shapes use N = number of trajectories,
nt = number of frames (``NT`` = 201 for every split), NX grid points and NFACE faces.

Datasets
--------
x         (NX,)            f8  grid, = constants.X
t         (nt,)            f8  frame times, t_k = k * DT
seed      (N,)             i8  per-trajectory index i used in SeedSequence([base_seed, split_id, i])
family    (N,)             u1  potential family id (constants.FAMILY_IDS)
Z         (N,)             f8  softened-atom charge in v0 = -Z/(|x|+1) + dv (NaN if the family has none)
E         (N,)             f8  uniform field; perturbation E*x switched on at t = 0+
A         (N,)             f8  max |dv| of the Fourier perturbation (NaN if the family has none)
v0        (N, NX)          f8  ground-state potential (t < 0)
n         (N, nt, NX)      f4  density n(x_i, t_k); sum(n) * DX = 1
j         (N, nt, NFACE)   f4  instantaneous current at face i+1/2 (between x_i and x_{i+1}),
                               positive = flow towards +x, consistent with the 13-pt stencil:
                               j_{i+1/2} = -DX * cumsum(dn/dt)[i], dn/dt = 2 Im(conj(psi) * H1 psi)
energy    (N,)             f8  <H1> for t > 0 (conserved)
eps0      (N,)             f8  ground-state energy of H0
gap0      (N,)             f8  eps1 - eps0 of H0
spec_eps  (N, N_SPEC)      f8  eigenvalues of H1 with the largest populations |<a|psi0>|^2 (sorted by energy)
spec_w    (N, N_SPEC)      f8  the corresponding populations
norm_err  (N,)             f8  max_k |sum(n_k) * DX - 1| computed in float64 before the cast to f4

Attributes
----------
schema_version, split, base_seed, nt, nx, dt, L, stencil, idea_version, git_hash, config_json, family_names

The full potential for t > 0 is ``potential(v0, E) = v0 + E * x``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from lqm import constants as C

logger = logging.getLogger(__name__)

# name -> (shape template, numpy dtype). Template symbols: "N", "nt", or ints.
SCHEMA: dict[str, tuple[tuple[Any, ...], str]] = {
    "x": ((C.NX,), "f8"),
    "t": (("nt",), "f8"),
    "seed": (("N",), "i8"),
    "family": (("N",), "u1"),
    "Z": (("N",), "f8"),
    "E": (("N",), "f8"),
    "A": (("N",), "f8"),
    "v0": (("N", C.NX), "f8"),
    "n": (("N", "nt", C.NX), "f4"),
    "j": (("N", "nt", C.NFACE), "f4"),
    "energy": (("N",), "f8"),
    "eps0": (("N",), "f8"),
    "gap0": (("N",), "f8"),
    "spec_eps": (("N", C.N_SPEC), "f8"),
    "spec_w": (("N", C.N_SPEC), "f8"),
    "norm_err": (("N",), "f8"),
}

REQUIRED_ATTRS: tuple[str, ...] = (
    "schema_version",
    "split",
    "base_seed",
    "nt",
    "nx",
    "dt",
    "L",
    "stencil",
    "idea_version",
    "git_hash",
    "config_json",
    "family_names",
)


def _expected_shape(template: tuple[Any, ...], n_traj: int, nt: int) -> tuple[int, ...]:
    return tuple(n_traj if s == "N" else nt if s == "nt" else int(s) for s in template)


def _validate(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    """Check names, shapes and dtypes of in-memory arrays. Returns (N, nt)."""
    missing = set(SCHEMA) - set(arrays)
    extra = set(arrays) - set(SCHEMA)
    if missing or extra:
        raise ValueError(f"schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    n_traj = int(np.shape(arrays["seed"])[0])
    nt = int(np.shape(arrays["t"])[0])
    for name, (template, dtype) in SCHEMA.items():
        a = np.asarray(arrays[name])
        exp = _expected_shape(template, n_traj, nt)
        if a.shape != exp:
            raise ValueError(f"{name}: shape {a.shape}, expected {exp}")
        if a.dtype != np.dtype(dtype):
            raise ValueError(f"{name}: dtype {a.dtype}, expected {np.dtype(dtype)}")
    return n_traj, nt


def write_split(path: str | Path, arrays: Mapping[str, np.ndarray], attrs: Mapping[str, Any]) -> None:
    """Write one split to HDF5 after validating it against ``SCHEMA``.

    Parameters
    ----------
    path : str or Path
        Output file (overwritten if it exists).
    arrays : mapping
        Every dataset in ``SCHEMA`` with the exact shape and dtype.
    attrs : mapping
        Every attribute in ``REQUIRED_ATTRS``. ``config_json`` may be a dict (serialised here).
    """
    n_traj, nt = _validate(arrays)
    missing = set(REQUIRED_ATTRS) - set(attrs)
    if missing:
        raise ValueError(f"missing attrs: {sorted(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name in SCHEMA:
            a = np.asarray(arrays[name])
            chunks = (1, *a.shape[1:]) if name in ("n", "j") and n_traj > 0 else None
            f.create_dataset(name, data=a, chunks=chunks)
        for key in REQUIRED_ATTRS:
            val = attrs[key]
            if key == "config_json" and not isinstance(val, str):
                val = json.dumps(val, sort_keys=True)
            if key == "family_names":
                val = np.array([str(s) for s in val], dtype=h5py.string_dtype())
            f.attrs[key] = val
    logger.info("wrote %s (N=%d, nt=%d)", path, n_traj, nt)


def load_split(
    path: str | Path,
    keys: Iterable[str] | None = None,
    device: Any = None,
) -> dict[str, Any]:
    """Load datasets (and attributes) from one split.

    Parameters
    ----------
    path : str or Path
        HDF5 file written by ``write_split``.
    keys : iterable of str, optional
        Datasets to load (default: all).
    device : torch device or str, optional
        If given, arrays are returned as torch tensors on this device (dtype preserved;
        u1 -> uint8, i8 -> int64). Otherwise numpy arrays.

    Returns
    -------
    dict
        ``{name: array}`` plus ``"attrs": dict`` of file attributes.
    """
    keys = list(SCHEMA) if keys is None else list(keys)
    unknown = set(keys) - set(SCHEMA)
    if unknown:
        raise KeyError(f"unknown datasets: {sorted(unknown)}")
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for k in keys:
            out[k] = f[k][...]
        attrs = {k: f.attrs[k] for k in f.attrs}
    if "family_names" in attrs:
        attrs["family_names"] = [s.decode() if isinstance(s, bytes) else str(s) for s in attrs["family_names"]]
    if "config_json" in attrs:
        attrs["config"] = json.loads(attrs["config_json"])
    if device is not None:
        import torch

        out = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in out.items()}
    out["attrs"] = attrs
    return out


def potential(v0: Any, E: Any, x: Any = None) -> Any:
    """Full potential for t > 0, v(x) = v0(x) + E * x.

    Works for numpy arrays and torch tensors with v0 of shape (..., NX) and E of shape (...).
    """
    if x is None:
        x = C.X
        if not isinstance(v0, np.ndarray):  # torch tensor
            import torch

            x = torch.tensor(C.X, dtype=v0.dtype, device=v0.device)  # copy: C.X is read-only
    return v0 + E[..., None] * x


def check_schema(path: str | Path) -> list[str]:
    """Return a list of schema violations in an existing file (empty list = valid)."""
    problems: list[str] = []
    with h5py.File(path, "r") as f:
        for key in REQUIRED_ATTRS:
            if key not in f.attrs:
                problems.append(f"missing attr {key}")
        if "seed" not in f or "t" not in f:
            return problems + ["missing seed or t"]
        n_traj, nt = f["seed"].shape[0], f["t"].shape[0]
        for name, (template, dtype) in SCHEMA.items():
            if name not in f:
                problems.append(f"missing dataset {name}")
                continue
            exp = _expected_shape(template, n_traj, nt)
            if f[name].shape != exp:
                problems.append(f"{name}: shape {f[name].shape}, expected {exp}")
            if f[name].dtype != np.dtype(dtype):
                problems.append(f"{name}: dtype {f[name].dtype}, expected {np.dtype(dtype)}")
        if "schema_version" in f.attrs and int(f.attrs["schema_version"]) != C.SCHEMA_VERSION:
            problems.append(f"schema_version {f.attrs['schema_version']} != {C.SCHEMA_VERSION}")
    return problems
