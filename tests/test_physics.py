"""Verification of the exact LQM-1e physics and of the generated splits.

Every tolerance here is the one specified in the project plan's verification section. The
cross-checks against iDEA are the ground truth for the Hamiltonian, the ground state and the
time propagation; the remaining tests check conservation laws, discrete continuity, parity,
frame-Nyquist sampling, rejection criteria, reproducibility and the schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ks_2samp

from lqm import constants as C
from lqm.io import check_schema, load_split, write_split
from lqm.physics import potentials as P
from lqm.physics.generate import _split_spec, default_config, generate_split, generate_trajectory
from lqm.physics.hamiltonian import hamiltonian, idea_hamiltonian, idea_system, kinetic_matrix, silence_idea
from lqm.physics.propagate import (
    current,
    density,
    density_rate,
    dipole,
    dipole_velocity,
    eigensystem,
    expectation_energy,
    ground_state,
    populations,
    simulate,
    wavefunction,
)
from lqm.utils import trajectory_rng

BASE_SEED = 20261123
N_SMALL = {"train": 32, "val": 8, "test": 8, "ood_field": 8, "ood_Z": 8, "ood_family": 9}
CONTINUITY_DELTA = 1e-4  # a.u.
CONTINUITY_FRAMES = np.array([0.0, 0.2, 1.0, 5.0, 20.0, 40.0])


@pytest.fixture(scope="session")
def config() -> dict:
    """The generation configuration used by every test."""
    return default_config()


@pytest.fixture(scope="session")
def splits(config) -> dict[str, dict[str, np.ndarray]]:
    """A small generated version of every split (schema arrays, not written to disk)."""
    return {s: generate_split(s, n, BASE_SEED, chunk_size=16, config=config)[0] for s, n in N_SMALL.items()}


@pytest.fixture(scope="session")
def exact(splits) -> dict[str, dict[str, np.ndarray]]:
    """float64 re-simulation of every small split (the stored arrays are float32)."""
    return {s: simulate(d["v0"], d["E"]) for s, d in splits.items()}


def _samples(splits: dict, n_each: int = 1) -> list[tuple[np.ndarray, float]]:
    """A few (v0, E) pairs spread over the splits, plus the in-distribution field extreme."""
    out = [(splits["train"]["v0"][i], float(splits["train"]["E"][i])) for i in range(n_each)]
    out.append((splits["train"]["v0"][0], 0.1))  # largest in-distribution |E|
    out.append((splits["ood_field"]["v0"][0], float(splits["ood_field"]["E"][0])))
    out.append((splits["ood_family"]["v0"][0], float(splits["ood_family"]["E"][0])))
    return out


# --- 1. Hamiltonian ----------------------------------------------------------------------


def test_kinetic_stencil_from_first_principles():
    """K = -1/2 d²/dx² with the 13-point central stencil, derived here independently of iDEA.

    The symmetric coefficients c_0..c_6 solve the moment conditions
    c_0 [m=0] + 2 sum_k c_k k^m = m! [m=2] for even m = 0..12, in exact rational arithmetic.
    Rows near the walls are truncated (zero wavefunction outside the box), as in iDEA.
    """
    from fractions import Fraction
    from math import factorial

    ms = range(0, 13, 2)
    a = [[Fraction(1 if m == 0 else 0)] + [Fraction(2 * k**m) for k in range(1, 7)] for m in ms]
    b = [Fraction(factorial(2) if m == 2 else 0) for m in ms]
    for col in range(7):  # Gauss-Jordan elimination in exact rationals
        piv = next(r for r in range(col, 7) if a[r][col] != 0)
        a[col], a[piv], b[col], b[piv] = a[piv], a[col], b[piv], b[col]
        for r in range(7):
            if r != col and a[r][col] != 0:
                f = a[r][col] / a[col][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col], strict=True)]
                b[r] -= f * b[col]
    c = [float(b[k] / a[k][k]) for k in range(7)]
    assert c[0] == pytest.approx(-5369 / 1800, rel=1e-15)  # known 12th-order central coefficient
    k_ref = np.zeros((C.NX, C.NX))
    for off in range(-6, 7):
        k_ref += np.diag(np.full(C.NX - abs(off), -0.5 * c[abs(off)] / C.DX**2), k=off)
    k = kinetic_matrix()
    assert np.abs(k - k_ref).max() < 1e-12 * np.abs(k_ref).max()


def test_hamiltonian_matches_idea(splits):
    """Check 1: H = K + diag(v) is iDEA's Hamiltonian to < 1e-12 for every family."""
    worst = 0.0
    for d in splits.values():
        for i in (0, 1, 2):
            v = d["v0"][i] + d["E"][i] * C.X
            worst = max(worst, float(np.abs(hamiltonian(v) - idea_hamiltonian(v)).max()))
    assert worst < 1e-12, f"max |H - H_iDEA| = {worst:.3e}"


def test_kinetic_matrix_properties():
    """The cached kinetic matrix is the symmetric, parity-even 13-point stencil."""
    k = kinetic_matrix()
    assert k.shape == (C.NX, C.NX) and not k.flags.writeable
    assert np.array_equal(k, k.T)
    assert np.array_equal(k, k[::-1, ::-1])  # needed for the parity test below
    assert np.count_nonzero(k[C.NX // 2]) == C.STENCIL
    assert np.linalg.eigvalsh(k).min() > 0.0  # positive definite with Dirichlet walls


# --- 2. Ground state ---------------------------------------------------------------------


def test_ground_state_matches_idea(splits):
    """Check 2: ground-state density and eps0 agree with iDEA's solve (1e-9 / 1e-10)."""
    import iDEA.methods.non_interacting as ni
    import iDEA.observables

    for v0 in splits["train"]["v0"][:3]:
        s = idea_system(v0)
        with silence_idea():
            state = ni.solve(s, k=0, silent=True)
        n_idea = iDEA.observables.density(s, state=state)
        eps, u0 = ground_state(v0[None])
        assert np.abs(n_idea - density(u0[0])).sum() * C.DX < 1e-9
        assert abs(float(state.up.energies[0]) - float(eps[0, 0])) < 1e-10


def test_ground_state_is_real_positive_and_normalised(exact, splits):
    """psi0 is sign-fixed non-negative and normalised; n(t=0) reproduces it."""
    for split, d in splits.items():
        _, u0 = ground_state(d["v0"])
        # The ground state is nodeless, so the sign fix makes it non-negative up to the
        # round-off of the eigensolver in the exponentially small tails.
        assert u0.min() > -1e-14, f"{split}: psi0 min = {u0.min():.3e}"
        assert u0.max() > 0.1
        np.testing.assert_allclose((u0**2).sum(axis=-1), 1.0, rtol=0, atol=1e-14)
        # n(t=0) goes through the spectral round trip U Uᵀ u0, so it agrees only to round-off.
        np.testing.assert_allclose(density(u0), exact[split]["n"][:, 0], rtol=1e-12, atol=1e-14)


# --- 3. Propagation against iDEA ---------------------------------------------------------


def _idea_density_evolution(v0: np.ndarray, E: float) -> np.ndarray:
    """Propagate with iDEA's single-body propagator and return n(t) on the frame grid."""
    import iDEA.methods.non_interacting as ni
    import iDEA.observables

    s = idea_system(v0)
    v_ptrb = np.tile(E * np.asarray(C.X), (C.NT, 1))
    with silence_idea():
        state = ni.solve(s, k=0, silent=True)
        evolution = ni.propagate(s, state, v_ptrb, np.asarray(C.T))
    return iDEA.observables.density(s, evolution=evolution)


def test_propagation_matches_idea_non_interacting(splits):
    """Check 3: exact propagation vs iDEA's expm stepping, max_k int|dn| dx < 1e-7.

    iDEA renormalises the orbital after every step, so only densities are compared.
    """
    errors = []
    for v0, E in _samples(splits, n_each=2):
        n_idea = _idea_density_evolution(v0, E)
        n_exact = simulate(v0[None], np.array([E]))["n"][0]
        errors.append(float(np.abs(n_idea - n_exact).sum(axis=-1).max() * C.DX))
    assert max(errors) < 1e-7, f"errors = {errors}"
    print(f"\nni.propagate: max_k int|dn| dx = {max(errors):.3e} over {len(errors)} samples")


@pytest.mark.slow
def test_propagation_matches_idea_interacting(splits):
    """Check 3b: same against the many-body propagator with a single electron."""
    import iDEA.methods.interacting as inter
    import iDEA.observables

    errors = []
    for v0, E in _samples(splits, n_each=1)[:2]:
        s = idea_system(v0)
        v_ptrb = np.tile(E * np.asarray(C.X), (C.NT, 1))
        with silence_idea():
            state = inter.solve(s, k=0)
            evolution = inter.propagate(s, state, v_ptrb, np.asarray(C.T))
        n_idea = iDEA.observables.density(s, evolution=evolution)
        n_exact = simulate(v0[None], np.array([E]))["n"][0]
        errors.append(float(np.abs(n_idea - n_exact).sum(axis=-1).max() * C.DX))
    assert max(errors) < 1e-7, f"errors = {errors}"
    print(f"\ninteracting.propagate: max_k int|dn| dx = {max(errors):.3e}")


# --- 4./5. Conservation ------------------------------------------------------------------


def test_norm_conserved(splits, exact):
    """Check 4: |int n dx - 1| < 1e-12 in float64 at every frame, < 1e-6 after the f4 cast."""
    for split, d in splits.items():
        err_f8 = np.abs(exact[split]["n"].sum(axis=-1) * C.DX - 1.0).max()
        err_f4 = np.abs(d["n"].astype("f8").sum(axis=-1) * C.DX - 1.0).max()
        assert err_f8 < 1e-12, f"{split}: float64 norm error {err_f8:.3e}"
        assert err_f4 < 1e-6, f"{split}: float32 norm error {err_f4:.3e}"
        assert d["norm_err"].max() < 1e-12
        np.testing.assert_allclose(d["norm_err"], exact[split]["norm_err"], rtol=0, atol=0)


def test_energy_conserved(splits, exact):
    """Check 5: <H1> is constant over all frames to 1e-10 relative, and equals the stored value."""
    worst = 0.0
    for d in splits.values():
        eps, u0 = ground_state(d["v0"])
        eps1, vecs = eigensystem(d["v0"] + d["E"][:, None] * C.X)
        c = populations(vecs, u0)
        u, hu = wavefunction(eps1, vecs, c, C.T)
        e = expectation_energy(u, hu)
        drift = np.abs(e - e.mean(axis=-1, keepdims=True)).max(axis=-1) / np.abs(e).mean(axis=-1)
        worst = max(worst, float(drift.max()))
        assert np.abs(e.mean(axis=-1) - d["energy"]).max() < 1e-12
    assert worst < 1e-10, f"relative energy drift {worst:.3e}"


# --- 6. Current and continuity -----------------------------------------------------------


def test_current_zero_at_t0_and_at_the_walls(splits, exact):
    """Check 6a: j(x, 0) = 0 and no flux through either hard wall (< 1e-12)."""
    for split, d in splits.items():
        j0 = np.abs(exact[split]["j"][:, 0, :]).max()
        assert j0 < 1e-12, f"{split}: |j(t=0)| = {j0:.3e}"
        assert np.abs(d["j"][:, 0, :]).max() == 0.0
        assert np.abs(exact[split]["wall_flux"]).max() < 1e-12


def test_continuity(splits):
    """Check 6b: the stored current satisfies dn/dt = -div j to 1e-6 of max|dn/dt|."""
    worst = 0.0
    for d in splits.values():
        eps, u0 = ground_state(d["v0"])
        eps1, vecs = eigensystem(d["v0"] + d["E"][:, None] * C.X)
        c = populations(vecs, u0)
        up, _ = wavefunction(eps1, vecs, c, CONTINUITY_FRAMES + CONTINUITY_DELTA)
        um, _ = wavefunction(eps1, vecs, c, CONTINUITY_FRAMES - CONTINUITY_DELTA)
        dndt_fd = (density(up) - density(um)) / (2.0 * CONTINUITY_DELTA)
        u, hu = wavefunction(eps1, vecs, c, CONTINUITY_FRAMES)
        j = current(density_rate(u, hu))
        div = np.diff(np.pad(j, [(0, 0), (0, 0), (1, 1)]), axis=-1) / C.DX
        worst = max(worst, float(np.abs(dndt_fd + div).max() / np.abs(dndt_fd).max()))
    assert worst < 1e-6, f"continuity residual {worst:.3e} of max|dn/dt|"
    print(f"\ncontinuity: residual = {worst:.3e} of max|dn/dt|")


def test_ehrenfest_dipole_velocity(splits):
    """Ehrenfest: dx * sum(j) equals d<x>/dt from a delta = 1e-4 central difference.

    By summation by parts this follows from continuity, so it is a consistency check of
    ``dipole_velocity`` rather than a physics check independent of the continuity test.
    """
    worst = 0.0
    for d in splits.values():
        eps, u0 = ground_state(d["v0"])
        eps1, vecs = eigensystem(d["v0"] + d["E"][:, None] * C.X)
        c = populations(vecs, u0)
        up, _ = wavefunction(eps1, vecs, c, CONTINUITY_FRAMES + CONTINUITY_DELTA)
        um, _ = wavefunction(eps1, vecs, c, CONTINUITY_FRAMES - CONTINUITY_DELTA)
        fd = (dipole(density(up)) - dipole(density(um))) / (2.0 * CONTINUITY_DELTA)
        u, hu = wavefunction(eps1, vecs, c, CONTINUITY_FRAMES)
        v_from_j = dipole_velocity(current(density_rate(u, hu)))
        worst = max(worst, float(np.abs(fd - v_from_j).max() / np.abs(v_from_j).max()))
    assert worst < 1e-6, f"Ehrenfest residual {worst:.3e}"


# --- 7. Parity ---------------------------------------------------------------------------


def test_parity(splits):
    """Check 7: (v0(-x), -E) reproduces the mirrored density to 1e-12."""
    worst = 0.0
    for d in splits.values():
        out = simulate(d["v0"], d["E"])
        pairs = [P.parity_partner(v, e) for v, e in zip(d["v0"], d["E"], strict=True)]
        v0_p, E_p = zip(*pairs, strict=True)
        out_p = simulate(np.array(v0_p), np.array(E_p))
        worst = max(worst, float(np.abs(out_p["n"] - out["n"][..., ::-1]).max()))
    assert worst < 1e-12, f"parity error {worst:.3e}"
    print(f"\nparity: max |n_mirror - n[::-1]| = {worst:.3e}")


# --- 8./9. Sampling quality --------------------------------------------------------------


def test_frame_nyquist_aliasing(exact):
    """Check 8: the population above pi/DT is < 1e-8 for every generated trajectory."""
    worst = max(float(out["alias"].max()) for out in exact.values())
    assert worst < 1e-8, f"aliased weight {worst:.3e}"
    print(f"\naliasing: max weight above pi/DT = {worst:.3e}")


def test_gap_and_wall_density(splits):
    """Check 9: gap0 > 1e-3 everywhere; in-distribution wall density < 1e-6."""
    for split, d in splits.items():
        assert d["gap0"].min() > 1e-3, f"{split}: min gap0 = {d['gap0'].min():.3e}"
        if split in ("train", "val", "test", "ood_family"):
            wall = d["n"][:, 0, [0, -1]].max()
            assert wall < 1e-6, f"{split}: ground-state wall density {wall:.3e}"


def test_wall_density_rejection_actually_rejects(config):
    """Negative control for ``max_wall_density``: an impossible bound must reject every draw."""
    spec = dict(_split_spec(config, "train"))
    spec["accept"] = {"min_gap": 1e-3, "max_wall_density": -1.0}
    with pytest.raises(RuntimeError, match="no acceptable potential"):
        P.sample(spec, 0, trajectory_rng(BASE_SEED, "train", 0), max_attempts=3)


def test_rejection_actually_rejects(config):
    """Potentials that violate the acceptance criteria are resampled, not stored."""
    spec = dict(_split_spec(config, "train"))
    spec["accept"] = {"min_gap": 1e9}  # impossible: every draw must be rejected
    with pytest.raises(RuntimeError, match="no acceptable potential"):
        P.sample(spec, 0, trajectory_rng(BASE_SEED, "train", 0), max_attempts=3)


def test_spectrum_is_sorted_and_dominant(splits, exact):
    """spec_eps is energy-sorted and the 16 kept levels carry essentially all the weight."""
    for split, d in splits.items():
        assert np.all(np.diff(d["spec_eps"], axis=-1) > 0)
        assert d["spec_w"].sum(axis=-1).min() > 0.999
        np.testing.assert_allclose(d["spec_w"][:, 0], exact[split]["spec_w"][:, 0], rtol=0, atol=0)


# --- 10. Reproducibility -----------------------------------------------------------------


def test_regeneration_is_bit_identical(splits, config):
    """Check 10a: regenerating a trajectory alone reproduces the stored arrays exactly.

    ``ood_family`` is included because its rejection loop and root-finding are the parts most
    at risk of drifting; Z and A are NaN there, hence ``equal_nan``.
    """
    keys = ("n", "j", "v0", "E", "Z", "A", "energy", "eps0", "gap0", "spec_eps", "spec_w")
    for split, count in (("train", 20), ("ood_family", N_SMALL["ood_family"])):
        d = splits[split]
        for i in range(count):
            one = generate_trajectory(split, int(d["seed"][i]), BASE_SEED, config)
            for key in keys:
                assert np.array_equal(one[key][0], d[key][i], equal_nan=True), f"{split}/{key} differs at {i}"


def test_split_streams_are_disjoint(splits):
    """Check 10b: the same index in two splits gives unrelated potentials."""
    for a, b in (("train", "val"), ("train", "test"), ("val", "test"), ("train", "ood_field")):
        n = min(N_SMALL[a], N_SMALL[b])
        assert np.abs(splits[a]["v0"][:n] - splits[b]["v0"][:n]).min() > 1e-6
        assert np.abs(splits[a]["Z"][:n] - splits[b]["Z"][:n]).min() > 1e-6


def test_chunking_does_not_change_results(config):
    """Chunk size is a performance knob only."""
    a, _ = generate_split("ood_family", 6, BASE_SEED, chunk_size=1, config=config)
    b, _ = generate_split("ood_family", 6, BASE_SEED, chunk_size=6, config=config)
    for key in a:
        equal_nan = np.issubdtype(a[key].dtype, np.floating)
        assert np.array_equal(a[key], b[key], equal_nan=equal_nan), key


# --- 11. Schema --------------------------------------------------------------------------


def test_written_file_passes_schema(tmp_path, config):
    """Check 11: generated splits write, validate and load through lqm.io."""
    arrays, attrs = generate_split("ood_Z", 4, BASE_SEED, chunk_size=2, config=config)
    path = tmp_path / "ood_Z.h5"
    write_split(path, arrays, attrs)
    assert check_schema(path) == []
    d = load_split(path)
    np.testing.assert_array_equal(d["n"], arrays["n"])
    np.testing.assert_array_equal(d["x"], C.X)
    assert d["attrs"]["split"] == "ood_Z"
    assert d["attrs"]["base_seed"] == BASE_SEED
    assert d["attrs"]["config"]["splits"]["ood_Z"]["z_intervals"] == [[0.5, 1.0], [2.0, 3.0]]
    assert d["attrs"]["idea_version"].count(".") >= 1
    assert d["attrs"]["family_names"] == list(C.FAMILIES)


def test_config_file_matches_default():
    """configs/data.json is exactly the committed default configuration."""
    path = Path(__file__).resolve().parents[1] / "configs" / "data.json"
    assert json.loads(path.read_text()) == json.loads(json.dumps(default_config()))


# --- 12. OOD splits change exactly one factor --------------------------------------------


def _sample_parameters(split: str, n: int, config: dict) -> dict[str, np.ndarray]:
    """Draw ``n`` potentials of a split and return their scalar parameters (no propagation)."""
    spec = _split_spec(config, split)
    rows = [P.sample(spec, i, trajectory_rng(BASE_SEED, split, i)) for i in range(n)]
    v0 = np.array([r.v0 for r in rows])
    eps, _ = ground_state(v0)
    return {
        "Z": np.array([r.Z for r in rows]),
        "A": np.array([r.A for r in rows]),
        "E": np.array([r.E for r in rows]),
        "abs_E": np.abs([r.E for r in rows]),
        "family": np.array([r.family for r in rows]),
        "min_v0": v0.min(axis=-1),
        "eps0": eps[:, 0],
    }


@pytest.fixture(scope="session")
def parameter_samples(config) -> dict[str, dict[str, np.ndarray]]:
    """Parameter draws for the distribution comparisons (400 per atom_fourier split)."""
    return {s: _sample_parameters(s, 400, config) for s in ("train", "ood_field", "ood_Z")}


def test_ood_field_changes_only_the_field(parameter_samples):
    """Check 12a: |E| moves out of distribution, Z and A keep the training distribution."""
    train, ood = parameter_samples["train"], parameter_samples["ood_field"]
    assert ood["abs_E"].min() >= 0.1 and ood["abs_E"].max() <= 0.2
    assert abs((ood["E"] > 0).mean() - 0.5) < 0.1  # the sign is drawn symmetrically
    assert ks_2samp(train["abs_E"], ood["abs_E"]).pvalue < 1e-10
    for factor in ("Z", "A", "min_v0", "eps0"):
        p = ks_2samp(train[factor], ood[factor]).pvalue
        assert p > 1e-3, f"{factor} distribution moved (KS p = {p:.3g})"


def test_ood_Z_changes_only_the_charge(parameter_samples):
    """Check 12b: Z moves out of distribution, |E| and A keep the training distribution."""
    train, ood = parameter_samples["train"], parameter_samples["ood_Z"]
    assert np.all((ood["Z"] >= 0.5) & (ood["Z"] <= 3.0))
    assert np.all((ood["Z"] < 1.0) | (ood["Z"] > 2.0))
    # Uniform on the union of [0.5, 1) and (2, 3] puts a third of the mass in the lower interval.
    assert abs((ood["Z"] < 1.0).mean() - 1 / 3) < 0.07
    assert ks_2samp(train["Z"], ood["Z"]).pvalue < 1e-10
    for factor in ("abs_E", "A"):
        p = ks_2samp(train[factor], ood[factor]).pvalue
        assert p > 1e-3, f"{factor} distribution moved (KS p = {p:.3g})"


def test_ood_family_changes_only_the_shape(splits, config):
    """Check 12c: balanced non-atomic families, NaN Z/A, and eps0/min(v0) inside training."""
    d = splits["ood_family"]
    counts = np.bincount(d["family"], minlength=len(C.FAMILIES))
    assert counts[C.FAMILY_IDS["atom_fourier"]] == 0
    assert set(counts[1:]) == {N_SMALL["ood_family"] // 3}
    assert np.isnan(d["Z"]).all() and np.isnan(d["A"]).all()
    # The binding match makes eps0 and min(v0) draws from the training marginal, so a few OOD
    # draws can fall just outside a finite training sample's range; allow 10% of its spread.
    # Equality in distribution is checked by the slow test below.
    train = _sample_parameters("train", 200, config)
    for factor, values in (("eps0", d["eps0"]), ("min_v0", d["v0"].min(axis=-1))):
        lo, hi = train[factor].min(), train[factor].max()
        margin = 0.1 * (hi - lo)
        assert values.min() >= lo - margin and values.max() <= hi + margin, f"{factor} outside training range"


@pytest.mark.slow
def test_ood_family_binding_matches_training_distribution(config):
    """The binding match makes eps0 and min(v0) draws from the *training* distribution."""
    train = _sample_parameters("train", 400, config)
    ood = _sample_parameters("ood_family", 120, config)
    for factor in ("eps0", "min_v0"):
        p = ks_2samp(train[factor], ood[factor]).pvalue
        assert p > 1e-3, f"{factor} distribution moved (KS p = {p:.3g})"
    assert ks_2samp(train["abs_E"], ood["abs_E"]).pvalue > 1e-3
    assert set(ood["family"]) == set(P.FAMILY_SHAPES)


def test_binding_match_is_exact(config):
    """``match_binding`` hits both eps0 and min(v0) of the reference potential."""
    spec = _split_spec(config, "train")
    rng = trajectory_rng(BASE_SEED, "ood_family", 0)
    v0_ref, _, _ = P.sample_atom_fourier(rng, spec)
    eps0_ref, binding_ref = P.binding_depth(v0_ref) + v0_ref.min(), P.binding_depth(v0_ref)
    for family in P.FAMILY_SHAPES:
        shape = P._SHAPE_SAMPLERS[family](rng, _split_spec(config, "ood_family")[family])
        v0 = P.match_binding(shape, eps0_ref, binding_ref)
        assert v0 is not None
        eps, _ = ground_state(v0[None])
        assert abs(float(eps[0, 0]) - eps0_ref) < 1e-10
        assert abs(float(v0.min()) - float(v0_ref.min())) < 1e-9
