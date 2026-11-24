"""Single-particle Hamiltonians on the frozen LQM-1e grid.

Atomic units throughout (ħ = mₑ = e = 1, energies in Hartree, lengths in Bohr).

The Hamiltonian must match iDEA's exactly, because the dataset is cross-checked against
``iDEA.methods.non_interacting``: H = K + diag(v), with K the 13-point finite-difference
representation of −½ ∂²/∂x² on the hard-wall grid ``constants.X`` (Dirichlet boundaries are
implied by truncating the stencil at the edges of the matrix). ``kinetic_matrix`` takes K
straight from ``iDEA.methods.non_interacting.kinetic_energy_operator`` and caches it, so the
stencil can never drift out of sync; ``idea_hamiltonian`` is the slow reference used in tests.
"""

from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import numpy as np

from lqm import constants as C

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def silence_idea() -> Iterator[io.StringIO]:
    """Redirect stdout into a buffer while inside the block.

    Several iDEA entry points (``solve``, ``interacting.solve``) print progress to stdout.

    Yields
    ------
    io.StringIO
        The buffer that captured stdout.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def idea_system(v: np.ndarray) -> Any:
    """Build the one-electron ``iDEA.system.System`` for an external potential.

    Parameters
    ----------
    v : ndarray, shape (NX,)
        External potential on ``constants.X`` (Hartree).

    Returns
    -------
    iDEA.system.System
        System with the frozen grid, no interaction, a single spin-up electron and the
        frozen 13-point stencil.
    """
    import iDEA.system

    return iDEA.system.System(
        np.array(C.X, dtype=float),
        np.array(v, dtype=float),
        np.zeros((C.NX, C.NX)),
        "u",
        stencil=C.STENCIL,
    )


@lru_cache(maxsize=1)
def kinetic_matrix() -> np.ndarray:
    """Return iDEA's 13-point kinetic-energy matrix on the frozen grid (read-only).

    Returns
    -------
    ndarray, shape (NX, NX)
        K = −½ D₂ in Hartree, symmetric, cached and flagged read-only.
    """
    import iDEA.methods.non_interacting as ni

    with silence_idea():
        k = ni.kinetic_energy_operator(idea_system(np.zeros(C.NX)))
    k = np.ascontiguousarray(k, dtype=float)
    k.flags.writeable = False
    logger.debug("built %d-point kinetic matrix, ||K|| = %.3f", C.STENCIL, np.abs(k).max())
    return k


def hamiltonian(v: np.ndarray) -> np.ndarray:
    """Build H = K + diag(v), vectorised over any number of leading batch axes.

    Parameters
    ----------
    v : ndarray, shape (..., NX)
        Potential(s) in Hartree.

    Returns
    -------
    ndarray, shape (..., NX, NX)
        Dense Hamiltonian(s) in Hartree, float64. Bit-identical to ``K + np.diag(v)``.
    """
    v = np.asarray(v, dtype=float)
    if v.shape[-1] != C.NX:
        raise ValueError(f"v must have last axis NX={C.NX}, got {v.shape}")
    h = np.empty((*v.shape, C.NX), dtype=float)
    h[...] = kinetic_matrix()
    i = np.arange(C.NX)
    h[..., i, i] += v
    return h


def idea_hamiltonian(v: np.ndarray) -> np.ndarray:
    """Reference Hamiltonian straight from iDEA (slow; used to test ``hamiltonian``).

    Parameters
    ----------
    v : ndarray, shape (NX,)
        Potential in Hartree.

    Returns
    -------
    ndarray, shape (NX, NX)
        ``iDEA.methods.non_interacting.hamiltonian(System(x, v, 0, "u"))[0]``.
    """
    import iDEA.methods.non_interacting as ni

    with silence_idea():
        return ni.hamiltonian(idea_system(v))[0]
