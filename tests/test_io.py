"""Tests for the frozen constants, HDF5 schema and synthetic data."""

import numpy as np
import pytest

from lqm import constants as C
from lqm.io import REQUIRED_ATTRS, SCHEMA, check_schema, load_split, potential, write_split
from lqm.testing import fake_split
from lqm.utils import trajectory_rng


def test_constants():
    assert C.X.shape == (C.NX,)
    assert np.isclose(C.DX, 20.0 / 127.0, rtol=0, atol=1e-15)
    assert C.NFACE == C.NX - 1
    assert np.isclose(C.T[-1], 40.0)
    with pytest.raises(ValueError):
        C.X[0] = 1.0  # read-only


def test_fake_split_roundtrip(tmp_path):
    path = fake_split(tmp_path / "fake.h5", n=5, nt=21, seed=3)
    assert check_schema(path) == []
    d = load_split(path)
    assert d["n"].shape == (5, 21, C.NX) and d["n"].dtype == np.float32
    assert d["j"].shape == (5, 21, C.NFACE)
    assert d["attrs"]["config"]["fake"] is True
    assert d["attrs"]["family_names"] == list(C.FAMILIES)
    # norm conservation (stored as f4)
    assert np.abs(d["n"].astype("f8").sum(-1) * C.DX - 1).max() < 1e-6
    assert d["norm_err"].max() < 1e-12


def test_fake_split_continuity(tmp_path):
    """Stored current must satisfy dn/dt = -(j_{i+1/2} - j_{i-1/2}) / DX with zero wall flux."""
    d = load_split(fake_split(tmp_path / "f.h5", n=3, nt=201, seed=1))
    n = d["n"].astype("f8")
    j = d["j"].astype("f8")
    jp = np.concatenate([np.zeros_like(j[..., :1]), j, np.zeros_like(j[..., :1])], axis=-1)
    div = (jp[..., 1:] - jp[..., :-1]) / C.DX
    # central difference in time vs instantaneous divergence at the midpoint frames
    dndt = (n[:, 2:] - n[:, :-2]) / (2 * C.DT)
    resid = np.abs(dndt + div[:, 1:-1]).max()
    assert resid < 2e-3 * np.abs(dndt).max() + 1e-6  # O(DT^2) central-difference error


def test_fake_split_deterministic(tmp_path):
    a = load_split(fake_split(tmp_path / "a.h5", n=4, nt=11, seed=7))
    b = load_split(fake_split(tmp_path / "b.h5", n=4, nt=11, seed=7))
    np.testing.assert_array_equal(a["n"], b["n"])


def test_write_split_rejects_bad_shapes(tmp_path):
    d = load_split(fake_split(tmp_path / "f.h5", n=2, nt=5))
    attrs = d.pop("attrs")
    attrs = {k: attrs[k] for k in REQUIRED_ATTRS}
    bad = dict(d)
    bad["n"] = d["n"].astype("f8")
    with pytest.raises(ValueError, match="dtype"):
        write_split(tmp_path / "bad.h5", bad, attrs)
    bad = dict(d)
    del bad["j"]
    with pytest.raises(ValueError, match="missing"):
        write_split(tmp_path / "bad.h5", bad, attrs)


def test_load_split_torch(tmp_path):
    torch = pytest.importorskip("torch")
    d = load_split(fake_split(tmp_path / "f.h5", n=2, nt=5), keys=["n", "E", "v0"], device="cpu")
    assert isinstance(d["n"], torch.Tensor) and d["n"].dtype == torch.float32
    v = potential(d["v0"], d["E"])
    assert v.shape == (2, C.NX)
    np.testing.assert_allclose(v.numpy(), d["v0"].numpy() + d["E"].numpy()[:, None] * C.X)


def test_potential_numpy():
    v0 = np.zeros((3, C.NX))
    E = np.array([0.1, -0.1, 0.0])
    v = potential(v0, E)
    np.testing.assert_allclose(v[0], 0.1 * C.X)


def test_trajectory_rng_independent():
    a = trajectory_rng(1, "train", 0).random(4)
    b = trajectory_rng(1, "val", 0).random(4)
    c = trajectory_rng(1, "train", 0).random(4)
    assert not np.allclose(a, b)
    np.testing.assert_array_equal(a, c)


def test_schema_covers_all_datasets():
    assert set(SCHEMA) >= {"n", "j", "v0", "E", "Z", "seed", "family"}
