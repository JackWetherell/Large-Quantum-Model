"""Schema-valid synthetic data for developing and testing downstream code (frozen interface).

``fake_split`` writes trajectories that are exact solutions of the time-dependent Schrödinger
equation for a particle in the hard-wall box with Hamiltonian H = sum_k E_k |phi_k><phi_k|,
where phi_k are the discrete sine modes on the grid (exactly orthonormal on the DST-I grid) and
E_k = (k pi / 2L)^2 / 2. The densities therefore conserve norm to machine precision and the
stored current satisfies the discrete continuity equation exactly.

The potential fields (v0 = 0, random E) are NOT consistent with these dynamics: the data are
for exercising pipelines (shapes, normalisers, heads, metrics), not for physics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lqm import constants as C
from lqm.io import write_split
from lqm.utils import git_hash

_N_MODES = 3


def _box_modes() -> tuple[np.ndarray, np.ndarray]:
    """Return (phi [k, NX], energies [k]) for k = 1.._N_MODES, with sum(phi_k^2) * DX = 1."""
    i = np.arange(C.NX)
    k = np.arange(1, _N_MODES + 1)
    phi = np.sqrt(2.0 / ((C.NX - 1) * C.DX)) * np.sin(np.pi * np.outer(k, i) / (C.NX - 1))
    energies = 0.5 * (k * np.pi / (2 * C.L)) ** 2
    return phi, energies


def fake_split(path: str | Path, n: int = 16, nt: int = C.NT, seed: int = 0, split: str = "train") -> Path:
    """Write ``n`` synthetic, schema-valid trajectories with ``nt`` frames to ``path``.

    Parameters
    ----------
    path : str or Path
        Output HDF5 file.
    n : int
        Number of trajectories.
    nt : int
        Number of frames (t_k = k * DT).
    seed : int
        RNG seed (fully determines the file contents).
    split : str
        Value of the ``split`` attribute.

    Returns
    -------
    Path
        The written file.
    """
    rng = np.random.default_rng(seed)
    phi, energies = _box_modes()
    t = C.DT * np.arange(nt)

    # Random superpositions dominated by the lowest mode.
    amp = np.abs(rng.normal(size=(n, _N_MODES))) * np.array([1.0, 0.5, 0.25])
    amp /= np.linalg.norm(amp, axis=1, keepdims=True)
    phase = rng.uniform(0, 2 * np.pi, size=(n, _N_MODES))
    c = amp * np.exp(1j * phase)  # (n, modes)

    # psi[b, k, x] = sum_m c[b,m] exp(-i E_m t_k) phi[m, x];  H psi uses E_m.
    ct = c[:, None, :] * np.exp(-1j * np.outer(t, energies))[None]  # (n, nt, modes)
    psi = ct @ phi
    hpsi = (ct * energies) @ phi
    dens = np.abs(psi) ** 2
    dndt = 2.0 * np.imag(np.conj(psi) * hpsi)
    cur = -C.DX * np.cumsum(dndt, axis=-1)[..., : C.NFACE]

    weights = np.abs(c) ** 2
    spec_eps = np.full((n, C.N_SPEC), np.nan)
    spec_w = np.zeros((n, C.N_SPEC))
    spec_eps[:, :_N_MODES] = energies
    spec_w[:, :_N_MODES] = weights

    arrays = {
        "x": np.array(C.X, dtype="f8"),
        "t": t.astype("f8"),
        "seed": np.arange(n, dtype="i8"),
        "family": np.zeros(n, dtype="u1"),
        "Z": np.ones(n, dtype="f8"),
        "E": rng.uniform(-0.1, 0.1, size=n).astype("f8"),
        "A": np.zeros(n, dtype="f8"),
        "v0": np.zeros((n, C.NX), dtype="f8"),
        "n": dens.astype("f4"),
        "j": cur.astype("f4"),
        "energy": (weights * energies).sum(axis=1).astype("f8"),
        "eps0": np.full(n, energies[0], dtype="f8"),
        "gap0": np.full(n, energies[1] - energies[0], dtype="f8"),
        "spec_eps": spec_eps,
        "spec_w": spec_w,
        "norm_err": np.abs(dens.sum(axis=-1) * C.DX - 1.0).max(axis=1).astype("f8"),
    }
    attrs = {
        "schema_version": C.SCHEMA_VERSION,
        "split": split,
        "base_seed": seed,
        "nt": nt,
        "nx": C.NX,
        "dt": C.DT,
        "L": C.L,
        "stencil": C.STENCIL,
        "idea_version": "n/a (synthetic)",
        "git_hash": git_hash(),
        "config_json": {"fake": True, "n": n, "nt": nt, "seed": seed},
        "family_names": C.FAMILIES,
    }
    write_split(path, arrays, attrs)
    return Path(path)
