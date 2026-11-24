"""Rollout, baselines and the end-to-end evaluation runner, on synthetic data."""

import json
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch

from lqm import constants as C
from lqm.eval import metrics as M
from lqm.eval.baselines import LinearExtrapolationStepper, OracleStepper, PersistenceStepper
from lqm.eval.rollout import rollout, teacher_forced
from lqm.eval.runner import RESULT_SCHEMA, evaluate, load_exact, load_results
from lqm.io import load_split
from lqm.model.stepper import Stepper
from lqm.testing import fake_split
from lqm.utils import REPO_ROOT


class HistoryStepper:
    """Toy model that uses every frame it is given and v: n̂ = 0.9 n_k + 0.1 mean(history) + 1e-3 v.

    Records each history it receives. ``predict_teacher_forced`` is the reference (loop) semantics.
    """

    name = "history"

    def __init__(self, context: int | None) -> None:
        self.context = context
        self.calls: list[torch.Tensor] = []

    def _f(self, hist: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return 0.9 * hist[:, -1] + 0.1 * hist.mean(dim=1) + 1e-3 * v

    def predict_next(self, n_hist, v):
        self.calls.append(n_hist.clone())
        return self._f(n_hist, v)

    def predict_teacher_forced(self, n, v):
        c = n.shape[1] if self.context is None else self.context
        return torch.stack([self._f(n[:, max(0, k + 1 - c) : k + 1], v) for k in range(n.shape[1])], dim=1)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    path = fake_split(tmp_path_factory.mktemp("data") / "test.h5", n=6, nt=C.NT, seed=5, split="test")
    d = load_exact(path)
    return torch.from_numpy(d["n"]), torch.from_numpy(d["v"])


def steppers():
    return [PersistenceStepper(), LinearExtrapolationStepper(), HistoryStepper(None), HistoryStepper(3)]


def test_baselines_satisfy_protocol():
    for s in [PersistenceStepper(), LinearExtrapolationStepper(), OracleStepper(np.zeros((1, 2, 4)), np.zeros((1, 4)))]:
        assert isinstance(s, Stepper)


def test_oracle_rollout_is_exact(data):
    n, v = data
    oracle = OracleStepper(n, v)
    idx = torch.tensor([4, 1, 3])  # subset in a different order: matched by (n_0, v), not by position
    pred = rollout(oracle, n[idx, 0], v[idx], C.NT - 1)
    assert torch.equal(pred, n[idx])
    assert np.all(M.l1(pred.numpy(), n[idx].numpy()) == 0.0)
    tf = teacher_forced(oracle, n[idx, :-1], v[idx])
    assert torch.equal(tf, n[idx, 1:])
    assert torch.isnan(teacher_forced(oracle, n[idx], v[idx])[:, -1]).all()  # no target beyond the data
    with pytest.raises(ValueError, match="no stored trajectory"):
        oracle.predict_next(n[:1, :1] * 1.001, v[:1])


def test_persistence_matches_numpy(data):
    n, v = data
    n_np = n.numpy()
    pred = rollout(PersistenceStepper(), n[:, 0], v, C.NT - 1).numpy()
    expected = np.array([[np.sum(np.abs(n_np[b, 0] - n_np[b, k])) * C.DX for k in range(C.NT)] for b in range(len(n))])
    assert np.abs(M.l1(pred, n_np) - expected).max() <= 1e-12
    assert expected[:, -1].min() > 1e-3  # the fake dynamics actually move
    tf = teacher_forced(PersistenceStepper(), n[:, :-1], v).numpy()
    expected_tf = np.abs(n_np[:, :-1] - n_np[:, 1:]).sum(axis=-1) * C.DX
    assert np.abs(M.l1(tf, n_np[:, 1:]) - expected_tf).max() <= 1e-12


def test_linear_extrapolation_conserves_norm(data):
    n, v = data
    lin = LinearExtrapolationStepper()
    pred = rollout(lin, n[:, 0], v, C.NT - 1)
    assert np.abs(M.norm_error(pred.numpy())).max() < 1e-12
    # From n_0 alone the first step is persistence, after which 2 n_0 - n_0 = n_0 exactly:
    # the free rollout of this baseline is identical to persistence.
    assert torch.equal(pred, rollout(PersistenceStepper(), n[:, 0], v, C.NT - 1))
    # A genuine 200-step extrapolation from two exact frames (n_k = n_0 + k (n_1 - n_0)) still
    # conserves the norm to rounding, although it drifts far from the data and goes negative.
    hist = n[:, :2].clone()
    for _ in range(C.NT - 1):
        hist = torch.cat([hist, lin.predict_next(hist[:, -2:], v)[:, None]], dim=1)
    assert np.abs(M.norm_error(hist.numpy())).max() < 1e-12
    assert M.negative_mass(hist[:, -1].numpy()).max() > 0
    tf = teacher_forced(lin, n, v)
    torch.testing.assert_close(tf[:, 1:], 2 * n[:, 1:] - n[:, :-1], rtol=0, atol=0)


def test_rollout_prefix_consistency(data):
    n, v = data
    for s in steppers():
        long = rollout(s, n[:, 0], v, 50)
        short = rollout(s, n[:, 0], v, 20)
        assert torch.equal(long[:, :21], short), s.name


def test_rollout_step1_equals_teacher_forced_pos0(data):
    n, v = data
    for s in steppers():
        r1 = rollout(s, n[:, 0], v, 1)[:, 1]
        tf0 = teacher_forced(s, n[:, :10], v)[:, 0]
        assert (r1 - tf0).abs().max().item() <= 1e-12, s.name


def test_rollout_respects_context(data):
    n, v = data
    n_steps = 10
    for context in (None, 1, 3):
        s = HistoryStepper(context)
        out = rollout(s, n[:, 0], v, n_steps)
        lengths = [h.shape[1] for h in s.calls]
        assert lengths == [k + 1 if context is None else min(k + 1, context) for k in range(n_steps)]
        for k, h in enumerate(s.calls):  # the history is the most recent frames of the rollout itself
            assert torch.equal(h, out[:, k + 1 - h.shape[1] : k + 1])
    # truncation changes the result for a history-dependent model
    assert not torch.equal(rollout(HistoryStepper(None), n[:, 0], v, 5), rollout(HistoryStepper(2), n[:, 0], v, 5))


def test_rollout_checks_stepper_output(data):
    n, v = data

    class Bad(PersistenceStepper):
        def predict_next(self, n_hist, v):
            return n_hist[:, -1].float()

    with pytest.raises(TypeError, match="float64"):
        rollout(Bad(), n[:, 0], v, 2)

    class BadShape(PersistenceStepper):
        def predict_next(self, n_hist, v):
            return n_hist[:, -1, :-1]

    with pytest.raises(ValueError, match="shape"):
        rollout(BadShape(), n[:, 0], v, 2)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rollout_on_gpu(data):
    n, v = data
    for s in [PersistenceStepper(), HistoryStepper(3), OracleStepper(n, v)]:
        gpu = rollout(s, n[:, 0].cuda(), v.cuda(), 20)
        assert gpu.device.type == "cuda" and gpu.dtype == torch.float64
        torch.testing.assert_close(gpu.cpu(), rollout(s, n[:, 0], v, 20), rtol=0, atol=1e-15)


def _strict_json(text: str) -> dict:
    def reject(token):
        raise ValueError(f"non-standard JSON token {token}")

    return json.loads(text, parse_constant=reject)


def _check_schema(path, n_traj, n_steps, n_save):
    sizes = {"N": n_traj, "T": n_steps + 1, "S": n_steps, "M": n_save, "NX": C.NX}
    with h5py.File(path, "r") as f:
        assert set(f) == set(RESULT_SCHEMA)
        for name, (template, dtype) in RESULT_SCHEMA.items():
            assert f[name].shape == tuple(sizes[s] for s in template), name
            assert f[name].dtype == np.dtype(dtype), name
        assert {"results_version", "model", "split", "git_hash", "config_json"} <= set(f.attrs)


def test_evaluate_end_to_end(tmp_path):
    test_path = fake_split(tmp_path / "data" / "test.h5", n=10, nt=C.NT, seed=3, split="test")
    ood_path = fake_split(tmp_path / "data" / "ood_field.h5", n=9, nt=C.NT, seed=4, split="ood_field")
    run = tmp_path / "results" / "run"

    # Persistence on the full split, several batches, a saved subset larger than one batch.
    summary = evaluate(PersistenceStepper(), test_path, run / "test.h5", batch_size=4, device="cpu", n_save=6)
    _check_schema(run / "test.h5", n_traj=10, n_steps=C.NT - 1, n_save=6)
    res = load_results(run / "test.h5")
    exact = load_exact(test_path)
    n = exact["n"]
    raw = load_split(test_path)
    assert res["attrs"]["model"] == "persistence" and res["attrs"]["config"]["n_steps"] == C.NT - 1
    np.testing.assert_array_equal(res["E"], raw["E"])
    np.testing.assert_array_equal(res["seed"], raw["seed"])
    np.testing.assert_array_equal(res["family"], raw["family"])
    np.testing.assert_allclose(res["t"], C.T[: C.NT], rtol=0, atol=0)
    assert np.abs(res["l1"] - np.abs(n[:, :1] - n).sum(-1) * C.DX).max() <= 1e-12
    assert np.abs(res["tf_l1"] - np.abs(n[:, :-1] - n[:, 1:]).sum(-1) * C.DX).max() <= 1e-12
    assert np.abs(res["norm_error"]).max() < 1e-12  # exact data renormalised in f64
    np.testing.assert_array_equal(res["negative_mass"], 0.0)
    np.testing.assert_allclose(res["dipole_true"], M.dipole(n), rtol=0, atol=1e-15)
    np.testing.assert_allclose(res["dipole_pred"], np.repeat(M.dipole(n[:, :1]), C.NT, axis=1), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(res["n_pred"], np.repeat(n[:6, :1], C.NT, axis=1))
    np.testing.assert_array_equal(res["save_index"], np.arange(6))

    # Summary: strict JSON, expected keys, numbers consistent with the per-trajectory arrays.
    s = _strict_json((run / "summary.json").read_text())["test"]
    assert set(s) >= {
        "rollout_l1",
        "time_to_threshold",
        "final_abs_norm_error",
        "final_negative_mass",
        "one_step_l1",
        "git_hash",
        "config",
        "n_traj",
        "n_nonfinite",
    }
    assert list(s["rollout_l1"]) == ["5", "10", "20", "40"]
    for k, tq in [("5", 25), ("40", 200)]:
        st = s["rollout_l1"][k]
        assert st["mean"] == pytest.approx(res["l1"][:, tq].mean(), rel=1e-12)
        assert st["median"] == pytest.approx(np.median(res["l1"][:, tq]), rel=1e-12)
        assert st["ci_lo"] <= st["mean"] <= st["ci_hi"]
    ttt = M.time_to_threshold(res["l1"])
    assert s["time_to_threshold"]["frac_censored"] == pytest.approx(np.isinf(ttt).mean())
    assert s["one_step_l1"]["mean"] == pytest.approx(res["tf_l1"].mean(), rel=1e-12)
    assert s["n_traj"] == 10 and s["n_nonfinite"] == 0
    assert s == summary

    # Oracle on a second split into the same run: shorter horizon, subset, merged summary.
    ex = load_exact(ood_path)
    evaluate(OracleStepper(ex["n"], ex["v"]), ood_path, run / "ood_field.h5", n_steps=50, batch_size=3, max_traj=7)
    _check_schema(run / "ood_field.h5", n_traj=7, n_steps=50, n_save=7)
    res = load_results(run / "ood_field.h5")
    assert np.all(res["l1"] == 0.0) and np.all(res["tf_l1"] == 0.0)
    np.testing.assert_array_equal(res["dipole_pred"], res["dipole_true"])
    both = _strict_json((run / "summary.json").read_text())
    assert set(both) == {"test", "ood_field"}
    so = both["ood_field"]
    assert list(so["rollout_l1"]) == ["5", "10"]
    assert so["time_to_threshold"]["median"] is None and so["time_to_threshold"]["frac_censored"] == 1.0
    assert so["config"]["split"] == "ood_field" and so["config"]["max_traj"] == 7


def test_evaluate_cli(tmp_path):
    fake_split(tmp_path / "data" / "test.h5", n=5, nt=26, seed=0, split="test")  # horizon t = 5
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"), "--model", "linear", "--splits", "test"]
    cmd += ["--data-dir", str(tmp_path / "data"), "--out", str(tmp_path / "out"), "--device", "cpu", "--max-traj", "3"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    _check_schema(tmp_path / "out" / "test.h5", n_traj=3, n_steps=25, n_save=3)
    s = _strict_json((tmp_path / "out" / "summary.json").read_text())["test"]
    assert s["config"]["model"] == "linear" and list(s["rollout_l1"]) == ["5"]
