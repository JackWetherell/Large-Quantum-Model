"""Potential samplers for the LQM-1e splits: pure functions of a ``numpy.random.Generator``.

Atomic units throughout (Hartree, Bohr). Every sampler draws from the generator in a fixed
order, so trajectory *i* of split *s* is fully determined by ``trajectory_rng(base_seed, s, i)``
and can be regenerated on its own.

Families
--------
``atom_fourier`` (training / in-distribution, and the only family in ``ood_field`` and ``ood_Z``)
    v0 = −Z/(|x|+1) + δv,  δv = Σ_{m=1..8} (a_m cos(mπx/L) + b_m sin(mπx/L)) / m,
    a_m, b_m ~ N(0,1), rescaled so max|δv| = A. The softened-Coulomb term pins a single well
    near x = 0; the 1/m roll-off makes δv smooth on the scale of the grid.
``harmonic`` (OOD)
    u = ½(x − x0)², x0 ~ U(−0.5, 0.5): a single, perfectly smooth parabolic well whose
    curvature is fixed by the binding match below (ω = √s ≈ 2(ε0 − min v0) ~ 0.75…1.5).
``double_well`` (OOD)
    u = −[(1+α) G(x−d) + (1−α) G(x+d)], G(y) = exp(−y²/2w²), with d ~ U(1.5, 3.5) (well
    centres at ±d), w ~ U(0.5, 1.0) and asymmetry α ~ U(−0.3, 0.3). Tunnelling splittings can
    be tiny, which is why near-degenerate draws are rejected.
``pure_fourier`` (OOD)
    u = the same Fourier series as δv above but with no atomic term and no fixed amplitude:
    a smooth multi-well landscape whose global minimum can sit anywhere in the box.

Matching OOD families to the training range
-------------------------------------------
Scaling a shape u by s and shifting it by a constant changes only two numbers: the binding
depth b = ε0 − min v0 (set by s) and the absolute energy offset (set by the shift). For every
``ood_family`` trajectory a *reference* ``atom_fourier`` potential is drawn from the same
generator; ``match_binding`` then solves b(s) = b_ref for s by bisection and shifts so that
ε0 = ε0_ref. Both min(v0) and ε0 therefore match an actual training draw exactly, so their
marginal distributions over the OOD split are identical to the training ones by construction
and only the *shape* of the potential is out of distribution.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as sla
from scipy.optimize import brentq

from lqm import constants as C
from lqm.physics.hamiltonian import hamiltonian
from lqm.physics.propagate import density, ground_state

logger = logging.getLogger(__name__)

FAMILY_SHAPES: tuple[str, ...] = ("double_well", "harmonic", "pure_fourier")
_MAX_SCALE: float = 2.0**20  # bisection bracket limit for the binding match


@dataclass(frozen=True)
class Potential:
    """One sampled trajectory specification.

    Attributes
    ----------
    v0 : ndarray, shape (NX,)
        Static potential for t < 0 (Hartree).
    E : float
        Uniform field switched on at t = 0 (Hartree/Bohr).
    Z : float
        Softened-atom charge, or NaN for families without one.
    A : float
        max|δv| of the Fourier perturbation, or NaN for families without one.
    family : str
        Family name, a key of ``constants.FAMILY_IDS``.
    attempts : int
        Number of draws needed before acceptance (1 = accepted immediately).
    """

    v0: np.ndarray
    E: float
    Z: float
    A: float
    family: str
    attempts: int = 1


def fourier_perturbation(rng: np.random.Generator, n_modes: int = 8) -> np.ndarray:
    """Draw δv = Σ_{m=1..M} (a_m cos(mπx/L) + b_m sin(mπx/L)) / m with a, b ~ N(0,1).

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness; consumes ``n_modes`` normals for a then ``n_modes`` for b.
    n_modes : int
        Number of Fourier modes M.

    Returns
    -------
    ndarray, shape (NX,)
        Unnormalised smooth perturbation (Hartree).
    """
    m = np.arange(1, n_modes + 1)
    a = rng.normal(size=n_modes)
    b = rng.normal(size=n_modes)
    arg = np.outer(m, np.pi * C.X / C.L)
    return ((a[:, None] * np.cos(arg) + b[:, None] * np.sin(arg)) / m[:, None]).sum(axis=0)


def sample_atom_fourier(rng: np.random.Generator, spec: Mapping[str, Any]) -> tuple[np.ndarray, float, float]:
    """Draw a softened-atom-plus-Fourier potential.

    Parameters
    ----------
    rng : numpy.random.Generator
        Draw order: Z, A, then the Fourier coefficients.
    spec : mapping
        Needs ``n_modes``, ``amp_range`` and either ``z_range`` (uniform interval) or
        ``z_intervals`` (list of intervals; Z is uniform on their union, i.e. an interval is
        picked with probability proportional to its width).

    Returns
    -------
    v0 : ndarray, shape (NX,)
        Potential (Hartree).
    Z : float
        Softened-atom charge.
    A : float
        max|δv| of the perturbation.
    """
    if "z_intervals" in spec:
        bounds = np.asarray(spec["z_intervals"], dtype=float)
        widths = bounds[:, 1] - bounds[:, 0]
        offsets = np.concatenate([[0.0], np.cumsum(widths)])
        u = rng.uniform(0.0, offsets[-1])
        k = min(int(np.searchsorted(offsets[1:], u, side="right")), len(widths) - 1)
        Z = float(bounds[k, 0] + u - offsets[k])
    else:
        Z = float(rng.uniform(*spec["z_range"]))
    A = float(rng.uniform(*spec["amp_range"]))
    dv = fourier_perturbation(rng, int(spec["n_modes"]))
    scale = A / np.abs(dv).max()
    return -Z / (np.abs(C.X) + 1.0) + scale * dv, Z, A


def sample_field(rng: np.random.Generator, spec: Mapping[str, Any]) -> float:
    """Draw the uniform field E.

    Parameters
    ----------
    rng : numpy.random.Generator
        Consumes one draw for ``field_range``, two for ``field_abs_range`` (magnitude, sign).
    spec : mapping
        Either ``field_range`` (E uniform on the signed interval, the training choice) or
        ``field_abs_range`` (|E| uniform on the interval with a random sign).

    Returns
    -------
    float
        Field strength in Hartree/Bohr.
    """
    if "field_abs_range" in spec:
        magnitude = float(rng.uniform(*spec["field_abs_range"]))
        return magnitude if rng.random() < 0.5 else -magnitude
    return float(rng.uniform(*spec["field_range"]))


def harmonic_shape(rng: np.random.Generator, spec: Mapping[str, Any]) -> np.ndarray:
    """Parabolic well ½(x − x0)², centre x0 ~ U(``center_range``). Scale is set by matching."""
    x0 = float(rng.uniform(*spec["center_range"]))
    return 0.5 * (C.X - x0) ** 2


def double_well_shape(rng: np.random.Generator, spec: Mapping[str, Any]) -> np.ndarray:
    """Two Gaussian wells at ±d with width w and asymmetry α; depth is set by matching."""
    d = float(rng.uniform(*spec["separation_range"]))
    w = float(rng.uniform(*spec["width_range"]))
    alpha = float(rng.uniform(*spec["asymmetry_range"]))
    g = lambda y: np.exp(-(y**2) / (2.0 * w**2))  # noqa: E731
    return -((1.0 + alpha) * g(C.X - d) + (1.0 - alpha) * g(C.X + d))


def pure_fourier_shape(rng: np.random.Generator, spec: Mapping[str, Any]) -> np.ndarray:
    """Bare Fourier series with no atomic term; amplitude is set by matching."""
    return fourier_perturbation(rng, int(spec["n_modes"]))


_SHAPE_SAMPLERS = {
    "harmonic": harmonic_shape,
    "double_well": double_well_shape,
    "pure_fourier": pure_fourier_shape,
}


def _ground_energy(v: np.ndarray) -> float:
    """Ground-state energy of H(v) only (cheaper than a full diagonalisation)."""
    return float(sla.eigh(hamiltonian(v), eigvals_only=True, subset_by_index=[0, 0])[0])


def binding_depth(v: np.ndarray) -> float:
    """b = ε0 − min(v): the ground-state energy measured from the bottom of the well (Hartree)."""
    return _ground_energy(v) - float(v.min())


def match_binding(shape: np.ndarray, eps0_ref: float, binding_ref: float) -> np.ndarray | None:
    """Scale and shift a shape so that ε0 and min(v0) match a reference potential exactly.

    Solves b(s) = ε0(s·u) − s·min(u) = ``binding_ref`` for the scale s > 0 by bisection, then
    adds the constant that puts ε0 at ``eps0_ref``. Because b is shift-invariant, this fixes
    min(v0) = ε0_ref − b_ref = min(v0_ref) as well.

    Parameters
    ----------
    shape : ndarray, shape (NX,)
        Unscaled potential shape u(x).
    eps0_ref : float
        Target ground-state energy (Hartree).
    binding_ref : float
        Target binding depth ε0 − min v0 (Hartree) > 0.

    Returns
    -------
    ndarray or None
        The matched potential, or None if no scale in (0, 2²⁰] brackets the target (the caller
        treats this as a rejected draw).
    """
    b = lambda s: binding_depth(s * shape) - binding_ref  # noqa: E731
    if b(0.0) > 0.0:  # free-particle zero-point energy already exceeds the target
        return None
    hi = 1.0
    while b(hi) < 0.0:
        hi *= 2.0
        if hi > _MAX_SCALE:
            logger.debug("binding match failed to bracket b_ref = %.3g", binding_ref)
            return None
    s = brentq(b, 0.0, hi, xtol=1e-13, rtol=1e-14)
    scaled = s * shape
    return scaled + (eps0_ref - _ground_energy(scaled))


def _draw(spec: Mapping[str, Any], index: int, rng: np.random.Generator) -> Potential | None:
    """Draw one candidate potential (no acceptance test). None = failed binding match."""
    v0_ref, Z, A = sample_atom_fourier(rng, spec)
    E = sample_field(rng, spec)
    if spec["family"] == "atom_fourier":
        return Potential(v0_ref, E, Z, A, "atom_fourier")

    # OOD family: the atom_fourier draw above is only the matching reference.
    families = list(spec["families"])
    family = families[index % len(families)]  # exactly balanced, no extra randomness
    shape = _SHAPE_SAMPLERS[family](rng, spec[family])
    v0 = match_binding(shape, _ground_energy(v0_ref), binding_depth(v0_ref))
    if v0 is None:
        return None
    return Potential(v0, E, float("nan"), float("nan"), family)


def sample(spec: Mapping[str, Any], index: int, rng: np.random.Generator, max_attempts: int = 64) -> Potential:
    """Draw an accepted potential for trajectory ``index`` of a split.

    Rejection criteria come from ``spec["accept"]``: ``min_gap`` (reject near-degenerate H0,
    where the ground state is not well defined and the dynamics are dominated by a slow
    tunnelling beat) and the optional ``max_wall_density`` (reject ground states that already
    touch the hard walls; applied to the OOD families, whose wells can sit anywhere, so that
    they stay comparable to training potentials, which are always centred).

    Parameters
    ----------
    spec : mapping
        Split specification (see ``lqm.physics.generate.default_config``).
    index : int
        Trajectory index within the split; also selects the family for ``ood_family``.
    rng : numpy.random.Generator
        From ``lqm.utils.trajectory_rng(base_seed, split, index)``.
    max_attempts : int
        Raise ``RuntimeError`` after this many rejected draws.

    Returns
    -------
    Potential
        Accepted potential, with the number of attempts recorded.
    """
    accept = spec.get("accept", {})
    for attempt in range(1, max_attempts + 1):
        candidate = _draw(spec, index, rng)
        if candidate is None:
            continue
        eps, u0 = ground_state(candidate.v0[None])
        gap = float(eps[0, 1] - eps[0, 0])
        if gap < accept.get("min_gap", 0.0):
            logger.debug("index %d: rejected, gap0 = %.3g", index, gap)
            continue
        if "max_wall_density" in accept:
            n0 = density(u0[0])
            if max(n0[0], n0[-1]) > accept["max_wall_density"]:
                logger.debug("index %d: rejected, wall density = %.3g", index, max(n0[0], n0[-1]))
                continue
        return Potential(candidate.v0, candidate.E, candidate.Z, candidate.A, candidate.family, attempt)
    raise RuntimeError(f"no acceptable potential for index {index} after {max_attempts} attempts")


def parity_partner(v0: np.ndarray, E: float) -> tuple[np.ndarray, float]:
    """Mirror image of a trajectory specification: (v0(−x), −E).

    The grid is symmetric and the 13-point kinetic matrix commutes with reversal, so the exact
    density of the partner is ``n[..., ::-1]``.

    Parameters
    ----------
    v0 : ndarray, shape (NX,)
        Potential.
    E : float
        Field.

    Returns
    -------
    tuple
        ``(v0[::-1], -E)``.
    """
    return np.ascontiguousarray(v0[::-1]), -E


def gap_and_wall_density(v0: np.ndarray) -> tuple[float, float]:
    """Diagnostics for one potential: (gap0, max ground-state density at the two walls).

    Parameters
    ----------
    v0 : ndarray, shape (NX,)
        Potential (Hartree).

    Returns
    -------
    gap0 : float
        ε1 − ε0 of H0 (Hartree).
    wall_density : float
        max(n0[0], n0[-1]) in 1/Bohr.
    """
    eps, vecs = np.linalg.eigh(hamiltonian(v0))
    n0 = density(vecs[:, 0])
    return float(eps[1] - eps[0]), float(max(n0[0], n0[-1]))
