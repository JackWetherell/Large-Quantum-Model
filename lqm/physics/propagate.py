"""Exact one-electron dynamics: ground state, propagation and observables (atomic units).

Protocol
--------
For t < 0 the electron sits in the ground state of H0 = H(v0). At t = 0 a uniform field is
switched on suddenly and then held constant, so for t > 0 the Hamiltonian is the *static*
H1 = H(v0 + E·x). The evolution is therefore exact and cheap: diagonalise H1 = U diag(ε) Uᵀ
once and write

    ψ(t) = U e^{−i ε t} Uᵀ ψ(0),      ψ(0) = ground state of H0,

which is spectrally exact at every frame (no time stepping, no Trotter error) and conserves the
norm and ⟨H1⟩ to machine precision.

Conventions
-----------
Internally wavefunctions are held in the **ℓ2 convention** u = ψ √dx, i.e. ``u @ u = 1`` and
``n = |u|² / dx``, because that is the convention of ``numpy.linalg.eigh``'s eigenvectors. Public
functions take and return ``u``; ``density`` does the dx bookkeeping. ψ(0) is real and is
sign-fixed to be non-negative (the ground state of a 1D Hamiltonian is nodeless).

The current is the *stencil-consistent* one. With a 13-point kinetic operator the textbook
expression Im(ψ* ∂ψ/∂x) does **not** satisfy the discrete continuity equation, so the current is
instead defined by integrating the exact density rate:

    dn/dt = 2 Im(ψ* H1 ψ),    j_{i+1/2} = −dx Σ_{m ≤ i} (dn/dt)_m,

which satisfies (dn/dt)_i = −(j_{i+1/2} − j_{i−1/2})/dx identically, with j = 0 at both walls
(the wall value vanishes because Σ_m (dn/dt)_m = 0 by norm conservation).
"""

from __future__ import annotations

import logging

import numpy as np

from lqm import constants as C
from lqm.physics.hamiltonian import hamiltonian

logger = logging.getLogger(__name__)


def ground_state(v0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalise H0 = H(v0) and return its spectrum and sign-fixed ground state.

    Parameters
    ----------
    v0 : ndarray, shape (..., NX)
        Static potential for t < 0 (Hartree).

    Returns
    -------
    eps : ndarray, shape (..., NX)
        All eigenvalues of H0 in ascending order (Hartree).
    u0 : ndarray, shape (..., NX)
        Ground state in the ℓ2 convention (``u0 @ u0 = 1``), sign-fixed so that u0 ≥ 0.
    """
    eps, vecs = np.linalg.eigh(hamiltonian(v0))
    u0 = vecs[..., 0]
    sign = np.where(u0.sum(axis=-1) < 0.0, -1.0, 1.0)
    return eps, u0 * sign[..., None]


def eigensystem(v1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalise the post-switch Hamiltonian H1 = H(v1).

    Parameters
    ----------
    v1 : ndarray, shape (..., NX)
        Potential for t > 0, ``v0 + E * x`` (Hartree).

    Returns
    -------
    eps : ndarray, shape (..., NX)
        Eigenvalues in ascending order (Hartree).
    vecs : ndarray, shape (..., NX, NX)
        Eigenvectors in columns, ℓ2-normalised.
    """
    return np.linalg.eigh(hamiltonian(v1))


def populations(vecs: np.ndarray, u0: np.ndarray) -> np.ndarray:
    """Expansion coefficients c_a = ⟨a|ψ(0)⟩ of the initial state in the H1 eigenbasis.

    Parameters
    ----------
    vecs : ndarray, shape (..., NX, NX)
        H1 eigenvectors in columns (ℓ2).
    u0 : ndarray, shape (..., NX)
        Initial state (ℓ2).

    Returns
    -------
    c : ndarray, shape (..., NX)
        Real coefficients with Σ_a c_a² = 1; the populations are c².
    """
    return np.einsum("...ia,...i->...a", vecs, u0)


def wavefunction(eps: np.ndarray, vecs: np.ndarray, c: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ψ(t) and H1ψ(t) at arbitrary times from the H1 spectral decomposition.

    Parameters
    ----------
    eps : ndarray, shape (B, NX)
        H1 eigenvalues (Hartree).
    vecs : ndarray, shape (B, NX, NX)
        H1 eigenvectors in columns.
    c : ndarray, shape (B, NX)
        Initial-state coefficients from ``populations``.
    t : ndarray, shape (nt,)
        Times (a.u.); t = 0 is the switch-on.

    Returns
    -------
    u : ndarray, shape (B, nt, NX), complex128
        ψ(t) in the ℓ2 convention.
    hu : ndarray, shape (B, nt, NX), complex128
        H1 ψ(t) in the same convention.
    """
    t = np.asarray(t, dtype=float)
    a = c[:, None, :] * np.exp(-1j * t[None, :, None] * eps[:, None, :])  # (B, nt, NX)
    vecs_t = np.swapaxes(vecs, -1, -2)
    return a @ vecs_t, (a * eps[:, None, :]) @ vecs_t


def density(u: np.ndarray) -> np.ndarray:
    """Density n = |ψ|² from an ℓ2-convention wavefunction, with Σ n dx = 1.

    Parameters
    ----------
    u : ndarray, shape (..., NX)
        Wavefunction(s) in the ℓ2 convention.

    Returns
    -------
    ndarray, shape (..., NX)
        Density in 1/Bohr.
    """
    return (u.real**2 + u.imag**2) / C.DX


def density_rate(u: np.ndarray, hu: np.ndarray) -> np.ndarray:
    """Exact dn/dt = 2 Im(ψ* H1 ψ) (1/(Bohr·a.u.)).

    Parameters
    ----------
    u, hu : ndarray, shape (..., NX)
        ψ and H1ψ in the ℓ2 convention (from ``wavefunction``).

    Returns
    -------
    ndarray, shape (..., NX)
        Instantaneous density rate.
    """
    return 2.0 * np.imag(np.conj(u) * hu) / C.DX


def current(dndt: np.ndarray) -> np.ndarray:
    """Stencil-consistent current on the NFACE interior faces, j_{i+1/2} = −dx Σ_{m≤i} dn/dt.

    Parameters
    ----------
    dndt : ndarray, shape (..., NX)
        Density rate from ``density_rate``.

    Returns
    -------
    ndarray, shape (..., NFACE)
        Current at faces i+1/2 (i = 0..NX-2), positive towards +x. The two wall faces are
        identically zero and are not stored.
    """
    return -C.DX * np.cumsum(dndt, axis=-1)[..., : C.NFACE]


def wall_flux(dndt: np.ndarray) -> np.ndarray:
    """Would-be current through the right wall, −dx Σ_m (dn/dt)_m (must vanish).

    Parameters
    ----------
    dndt : ndarray, shape (..., NX)
        Density rate.

    Returns
    -------
    ndarray, shape (...)
        Residual flux at the hard wall; zero up to round-off for exact dynamics.
    """
    return -C.DX * np.sum(dndt, axis=-1)


def expectation_energy(u: np.ndarray, hu: np.ndarray) -> np.ndarray:
    """⟨ψ|H1|ψ⟩ in Hartree, frame by frame (constant for the sudden-switch protocol).

    Parameters
    ----------
    u, hu : ndarray, shape (..., NX)
        ψ and H1ψ in the ℓ2 convention.

    Returns
    -------
    ndarray, shape (...)
        Expectation value of H1.
    """
    return np.real(np.sum(np.conj(u) * hu, axis=-1))


def spectrum(eps: np.ndarray, w: np.ndarray, n_spec: int = C.N_SPEC) -> tuple[np.ndarray, np.ndarray]:
    """The ``n_spec`` most populated H1 levels, returned in ascending energy order.

    Parameters
    ----------
    eps : ndarray, shape (B, NX)
        H1 eigenvalues (Hartree, ascending).
    w : ndarray, shape (B, NX)
        Populations |⟨a|ψ(0)⟩|².
    n_spec : int
        How many levels to keep.

    Returns
    -------
    spec_eps, spec_w : ndarray, shape (B, n_spec)
        Energies and populations of the most populated levels, sorted by energy.
    """
    idx = np.argsort(-w, axis=-1, kind="stable")[:, :n_spec]
    idx = np.sort(idx, axis=-1)  # eps is ascending, so sorting indices sorts by energy
    return np.take_along_axis(eps, idx, axis=-1), np.take_along_axis(w, idx, axis=-1)


def aliasing_weight(eps: np.ndarray, w: np.ndarray, dt: float = C.DT) -> np.ndarray:
    """Population above the frame-Nyquist limit, Σ_a w_a [ε_a − ε_min > π/dt].

    Frames sample the dynamics at spacing ``dt``, so a beat frequency ω = ε_a − ε_b above
    π/dt is aliased. Bounding the total population of levels further than π/dt above the
    lowest H1 level bounds every beat frequency that carries weight.

    Parameters
    ----------
    eps : ndarray, shape (B, NX)
        H1 eigenvalues (Hartree, ascending).
    w : ndarray, shape (B, NX)
        Populations.
    dt : float
        Frame spacing (a.u.).

    Returns
    -------
    ndarray, shape (B,)
        Aliased population weight (dimensionless).
    """
    return np.where(eps - eps[:, :1] > np.pi / dt, w, 0.0).sum(axis=-1)


def dipole(n: np.ndarray) -> np.ndarray:
    """⟨x⟩(t) = Σ_i x_i n_i dx in Bohr.

    Parameters
    ----------
    n : ndarray, shape (..., NX)
        Density.

    Returns
    -------
    ndarray, shape (...)
        Dipole position.
    """
    return (n * C.X).sum(axis=-1) * C.DX


def dipole_velocity(j: np.ndarray) -> np.ndarray:
    """d⟨x⟩/dt = dx Σ_faces j, the Ehrenfest velocity implied by the stored current.

    Summation by parts of d⟨x⟩/dt = Σ_i x_i (dn/dt)_i dx with (dn/dt)_i = −Δj_i/dx and zero
    wall current gives exactly dx Σ_i j_{i+1/2}.

    Parameters
    ----------
    j : ndarray, shape (..., NFACE)
        Current from ``current``.

    Returns
    -------
    ndarray, shape (...)
        d⟨x⟩/dt in Bohr per a.u. of time.
    """
    return j.sum(axis=-1) * C.DX


def simulate(
    v0: np.ndarray,
    E: np.ndarray,
    t: np.ndarray = C.T,
    n_spec: int = C.N_SPEC,
) -> dict[str, np.ndarray]:
    """Run the full exact protocol for a batch of trajectories, in float64.

    Parameters
    ----------
    v0 : ndarray, shape (B, NX)
        Static potentials for t < 0 (Hartree).
    E : ndarray, shape (B,)
        Uniform field switched on at t = 0; the potential for t > 0 is ``v0 + E * x``.
    t : ndarray, shape (nt,)
        Frame times (a.u.).
    n_spec : int
        Number of most-populated H1 levels to report.

    Returns
    -------
    dict of ndarray
        ``n`` (B, nt, NX) density; ``j`` (B, nt, NFACE) current; ``energy`` (B,) ⟨H1⟩;
        ``eps0``, ``gap0`` (B,) ground-state energy and gap of H0; ``spec_eps``, ``spec_w``
        (B, n_spec); ``norm_err`` (B,) max_k |Σ n dx − 1|; ``alias`` (B,) aliased weight;
        ``wall_flux`` (B, nt) residual flux at the wall; ``energy_drift`` (B,) max frame-to-frame
        deviation of ⟨H1⟩ from its mean, relative to |⟨H1⟩|. All float64.
    """
    v0 = np.atleast_2d(np.asarray(v0, dtype=float))
    E = np.atleast_1d(np.asarray(E, dtype=float))
    if v0.ndim != 2 or E.shape != v0.shape[:1]:
        raise ValueError(f"expected v0 (B, NX) and E (B,), got {v0.shape} and {E.shape}")

    eps_h0, u0 = ground_state(v0)
    eps, vecs = eigensystem(v0 + E[:, None] * C.X)
    c = populations(vecs, u0)
    u, hu = wavefunction(eps, vecs, c, t)

    n = density(u)
    dndt = density_rate(u, hu)
    w = c**2
    e_frames = expectation_energy(u, hu)
    spec_eps, spec_w = spectrum(eps, w, n_spec)
    return {
        "n": n,
        "j": current(dndt),
        "energy": (w * eps).sum(axis=-1),
        "eps0": eps_h0[:, 0],
        "gap0": eps_h0[:, 1] - eps_h0[:, 0],
        "spec_eps": spec_eps,
        "spec_w": spec_w,
        "norm_err": np.abs(n.sum(axis=-1) * C.DX - 1.0).max(axis=-1),
        "alias": aliasing_weight(eps, w),
        "wall_flux": wall_flux(dndt),
        "energy_drift": np.abs(e_frames - e_frames.mean(axis=-1, keepdims=True)).max(axis=-1)
        / np.abs(e_frames).mean(axis=-1),
    }
