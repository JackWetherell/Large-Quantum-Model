"""Metrics on analytic cases, and bootstrap reproducibility / coverage."""

import numpy as np
import pytest
from scipy.special import erf

from lqm import constants as C
from lqm.eval import metrics as M


def gaussian(mu: float, sigma: float = 1.0) -> np.ndarray:
    """Analytically normalised Gaussian on the grid (tails at the walls are < 1e-20)."""
    return np.exp(-((C.X - mu) ** 2) / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)


def test_l1_shifted_gaussians():
    # ∫|g(x - a) - g(x - b)| dx = 2 erf(|a - b| / (2√2 σ)); the kink at the crossing limits the
    # Riemann sum to O(dx²) accuracy.
    for a, b in [(-0.5, 0.5), (0.0, 0.3), (-1.0, 2.0)]:
        exact = 2 * erf(abs(a - b) / (2 * np.sqrt(2)))
        assert M.l1(gaussian(a), gaussian(b)) == pytest.approx(exact, abs=2e-3)
    assert M.l1(gaussian(0.2), gaussian(0.2)) == 0.0
    assert M.l1(gaussian(-5, 0.3), gaussian(5, 0.3)) == pytest.approx(2.0, abs=1e-12)  # disjoint


def test_metrics_broadcast_and_dx():
    n = np.stack([gaussian(m) for m in (-1.0, 0.0, 1.0)])[None]  # [1, 3, NX]
    assert M.l1(n, n[:, ::-1]).shape == (1, 3)
    assert M.norm_error(n).shape == (1, 3)
    np.testing.assert_allclose(M.l1(n, 0 * n, dx=2 * C.DX), 2 * M.l1(n, 0 * n), rtol=1e-15)


def test_rel_l2():
    n = gaussian(0.3)
    assert M.rel_l2(1.01 * n, n) == pytest.approx(0.01, rel=1e-12)
    assert M.rel_l2(n, n) == 0.0


def test_norm_and_dipole_of_gaussian():
    # The grid sum of a smooth, decaying function is spectrally accurate.
    for mu in (-2.0, 0.0, 1.5):
        assert abs(M.norm_error(gaussian(mu))) < 1e-12
        assert M.dipole(gaussian(mu)) == pytest.approx(mu, abs=1e-12)


def test_negative_mass():
    # The sign change sits at x ≈ 0.52, 11.6σ from the negative bump: tails there are ~1e-30.
    n = gaussian(-3.0, 0.3) - 0.25 * gaussian(4.0, 0.3)
    assert M.negative_mass(n) == pytest.approx(0.25, abs=1e-12)
    assert M.norm_error(n) == pytest.approx(-0.25, abs=1e-12)
    assert M.negative_mass(gaussian(0.0)) == 0.0


def test_dipole_spectrum_peak():
    dt, omega0 = C.DT, 0.45  # Ha; a typical dipole period ~ 14 a.u.
    t = C.T
    d = 0.3 + 0.2 * np.cos(omega0 * t)
    omega, p = M.dipole_spectrum(d, dt=dt, n_fft=8192)
    assert omega.shape == p.shape == (8192 // 2 + 1,)
    assert abs(omega[np.argmax(p)] - omega0) < 2 * np.pi / (8192 * dt)
    # mean removal: a static offset (field-shifted equilibrium) does not change the spectrum
    np.testing.assert_allclose(M.dipole_spectrum(d - 1.3, dt=dt, n_fft=8192)[1], p, rtol=0, atol=1e-12 * p.max())
    # batched input, default length
    omega, p2 = M.dipole_spectrum(np.stack([d, 2 * d]))
    assert p2.shape == (2, len(t) // 2 + 1)
    np.testing.assert_allclose(p2[1], 4 * p2[0], rtol=1e-12)


def test_time_to_threshold():
    t = C.T
    err = np.stack([0.0101 * t, np.full_like(t, 0.05), 0.0101 * t])  # [3, T]
    err[2, 30] = np.nan  # a diverged step counts as a crossing
    ttt = M.time_to_threshold(err, thr=0.1)
    assert ttt[0] == pytest.approx(10.0, abs=1e-12)  # 0.0101 t > 0.1 for t > 9.90
    assert np.isinf(ttt[1])  # never crosses: censored
    assert ttt[2] == pytest.approx(30 * C.DT, abs=1e-12)
    assert np.isinf(M.time_to_threshold(np.zeros(5)))
    # strict inequality: equal to the threshold is not a crossing
    assert M.time_to_threshold(np.array([0.0, 0.5, 0.75]), thr=0.5, dt=1.0) == 2.0


def test_bootstrap_reproducible_and_consistent():
    rng = np.random.default_rng(1)
    x = rng.exponential(size=200)
    a = M.bootstrap_ci(x, n_boot=500, seed=3)
    b = M.bootstrap_ci(x, n_boot=500, seed=3)
    c = M.bootstrap_ci(x, n_boot=500, seed=4)
    assert a == b
    assert a.lo != c.lo
    assert a.mean == pytest.approx(x.mean(), rel=1e-14)
    assert a.lo < a.mean < a.hi
    # 2-D input: each column equals the 1-D bootstrap with the same seed (same resamples).
    x2 = np.stack([x, 2 * x + 1], axis=1)
    ci2 = M.bootstrap_ci(x2, n_boot=500, seed=3)
    assert ci2.mean.shape == (2,)
    assert ci2.lo[0] == pytest.approx(a.lo, rel=1e-12)
    assert ci2.hi[1] == pytest.approx(2 * a.hi + 1, rel=1e-12)


def test_bootstrap_coverage():
    """95% percentile bootstrap covers the true mean ~95% of the time (n = 100, Gaussian)."""
    rng = np.random.default_rng(0)
    n_rep, mu = 400, 1.3
    hits = 0
    for i in range(n_rep):
        ci = M.bootstrap_ci(rng.normal(mu, 2.0, size=100), n_boot=500, seed=i)
        hits += ci.lo <= mu <= ci.hi
    coverage = hits / n_rep
    assert 0.90 <= coverage <= 0.98, coverage  # binomial s.d. ≈ 0.011 at 0.95


def test_paired_bootstrap():
    rng = np.random.default_rng(2)
    base = rng.lognormal(sigma=1.0, size=300)  # large shared trajectory-to-trajectory variation
    a = base + 0.05 + 0.01 * rng.normal(size=300)
    b = base
    paired = M.paired_bootstrap_ci(a, b, seed=0)
    assert paired.lo < 0.05 < paired.hi
    assert paired.lo > 0  # the difference is resolved ...
    ua, ub = M.bootstrap_ci(a, seed=0), M.bootstrap_ci(b, seed=1)
    assert ua.lo < ub.hi  # ... although the unpaired intervals overlap
    exact_shift = M.paired_bootstrap_ci(b + 0.1, b)
    assert exact_shift.lo == pytest.approx(0.1, abs=1e-12) and exact_shift.hi == pytest.approx(0.1, abs=1e-12)
    with pytest.raises(ValueError):
        M.paired_bootstrap_ci(a, b[:-1])
