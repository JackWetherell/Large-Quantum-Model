"""Training pipeline: normaliser, augmentation, schedule, losses, CLI artefacts, overfit sanity."""

import csv
import importlib.util
import json
import math

import numpy as np
import pytest
import torch

from lqm import constants as C
from lqm.io import load_split, potential
from lqm.model.checkpoint import build_model, load_stepper
from lqm.model.heads import HEADS
from lqm.model.normaliser import Normaliser, compute_normaliser
from lqm.testing import fake_split
from lqm.train.data import TrajectoryData, epoch_batches, parity_flip
from lqm.train.loop import LOG_COLUMNS, TrainConfig, evaluate, lr_at, train
from lqm.train.losses import l1_per_frame, pushforward_loss, scaled_mse, teacher_forced_loss
from lqm.utils import REPO_ROOT, set_global_seed

CUDA = torch.cuda.is_available()


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    path = fake_split(tmp_path_factory.mktemp("data") / "train.h5", n=6, seed=11)
    return path, TrajectoryData.load(path, "cpu")


def test_normaliser_matches_numpy(fake_data):
    path, data = fake_data
    d = load_split(path)
    n = d["n"].astype(np.float64)
    v = d["v0"] + d["E"][:, None] * C.X
    dn = np.diff(n, axis=1)
    J = -(C.DX / C.DT) * np.cumsum(dn, axis=-1)[..., : C.NFACE]
    expected = {
        "sigma_n": n.std(),
        "sigma_v": (v - v.mean(-1, keepdims=True)).std(),
        "sigma_delta": dn.std(),
        "sigma_J": J.std(),
    }
    for chunk_size in (4, 256):  # 2 chunks (exercises accumulation) and 1 chunk
        got = compute_normaliser(data.n, data.v, chunk_size=chunk_size).to_dict()
        for k, val in expected.items():
            assert math.isclose(got[k], val, rel_tol=1e-12), (k, got[k], val)
    assert Normaliser.from_dict(got).to_dict() == got


def test_parity_flip(fake_data):
    path, data = fake_data
    d = load_split(path)
    torch.manual_seed(0)
    n, v = parity_flip(data.n, data.v, p=1.0)
    assert torch.equal(n, data.n.flip(-1))
    # mirroring v is the same as mirroring v0 and reversing the field
    v_ref = potential(torch.from_numpy(d["v0"][:, ::-1].copy()), torch.from_numpy(-d["E"]))
    assert torch.allclose(v, v_ref, rtol=0, atol=1e-14)
    assert parity_flip(data.n, data.v, p=0.0)[0] is data.n
    n_half, _ = parity_flip(data.n, data.v, p=0.5)
    flipped = [torch.equal(n_half[i], data.n[i].flip(-1)) for i in range(len(data))]
    kept = [torch.equal(n_half[i], data.n[i]) for i in range(len(data))]
    assert all(f or k for f, k in zip(flipped, kept, strict=True))


def test_epoch_batches_cover_each_epoch():
    torch.manual_seed(0)
    it = epoch_batches(10, 3, "cpu")
    first = [next(it) for _ in range(3)]
    assert [e for e, _ in first] == [0, 0, 0]
    assert len(torch.cat([i for _, i in first]).unique()) == 9  # one incomplete batch dropped
    assert next(it)[0] == 1


def test_lr_schedule():
    cfg = TrainConfig(lr=1e-3, warmup_steps=10)
    lrs = [lr_at(s, cfg, 110) for s in range(110)]
    assert math.isclose(lrs[9], 1e-3) and lrs[0] == pytest.approx(1e-4)
    assert all(a <= b for a, b in zip(lrs[:10], lrs[1:10], strict=False))
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:], strict=False))
    assert lrs[60] == pytest.approx(0.5e-3) and lr_at(110, cfg, 110) == 0.0


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("model_type", ["gpt", "mlp"])
def test_initial_loss_is_persistence(fake_data, head, model_type):
    _, data = fake_data
    nm = compute_normaliser(data.n, data.v)
    model = build_model(model_type, head, {"n_layers": 2}, nm)
    loss = teacher_forced_loss(model, data.n, data.v).item()
    persistence = scaled_mse(data.n[:, :-1], data.n[:, 1:], nm.sigma_delta).item()
    if head == "softmax":  # renormalisation of float32-stored frames: ~1e-9 relative
        assert math.isclose(loss, persistence, rel_tol=1e-6)
    else:
        assert loss == persistence
    metrics = evaluate(model, data, batch_size=4)
    assert metrics["val_loss"] == pytest.approx(persistence, rel=1e-6)
    assert metrics["val_l1"] == pytest.approx(metrics["persistence_l1"], rel=1e-6)


def test_pushforward_inputs_are_shifted_predictions(fake_data):
    """At initialisation pass 1 is persistence, so pass 2 sees [n_0, n_0, n_1, ..., n_{T-3}]."""
    _, data = fake_data
    nm = compute_normaliser(data.n, data.v)
    model = build_model("gpt", "unconstrained", {"n_layers": 2}, nm)
    n = data.n.double()
    shifted = torch.cat([n[:, :1], n[:, :-2]], dim=1)
    loss = pushforward_loss(model, data.n, data.v)
    assert loss.item() == scaled_mse(shifted, n[:, 1:], nm.sigma_delta).item()
    loss.backward()
    assert model.head.proj.weight.grad.abs().sum() > 0


def test_early_stopping(fake_data, tmp_path):
    """With lr = 0 the val loss never improves on step 0, so training stops after `patience` evals."""
    _, data = fake_data
    model = build_model("mlp", "flux", {"d_model": 32, "n_layers": 1}, compute_normaliser(data.n, data.v))
    cfg = TrainConfig(batch_size=2, lr=0.0, warmup_steps=1, max_steps=100, eval_every=2, patience=3)
    summary = train(model, data, data, cfg, tmp_path)
    assert summary["stopped_early"] and summary["final_step"] == 6 and summary["best_step"] == 0
    assert [int(r["step"]) for r in csv.DictReader((tmp_path / "log.csv").open())] == [0, 2, 4, 6]


def _load_train_script():
    spec = importlib.util.spec_from_file_location("train_script", REPO_ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model_type", ["gpt", "mlp"])
def test_train_cli_writes_artefacts(tmp_path, model_type):
    data_dir = tmp_path / "data"
    fake_split(data_dir / "train.h5", n=12, seed=1, split="train")
    fake_split(data_dir / "val.h5", n=4, seed=2, split="val")
    model_cfg = {"d_model": 32, "n_layers": 1} | ({"n_heads": 4} if model_type == "gpt" else {})
    cfg = json.loads((REPO_ROOT / "configs" / f"train_{model_type}.json").read_text())
    cfg["model"].update(model_cfg)
    cfg["train"].update({"batch_size": 4, "eval_every": 3, "warmup_steps": 2, "pushforward_frac": 0.5})
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    argv = ["--config", str(cfg_path), "--head", "softmax", "--seed", "3", "--data-dir", str(data_dir)]
    argv += ["--runs-dir", str(tmp_path / "runs"), "--run-name", "smoke", "--max-steps", "6"]
    argv += ["--device", "cuda" if CUDA else "cpu"]
    script = _load_train_script()
    summary = script.main(argv)
    run = tmp_path / "runs" / "smoke"
    assert {"config.json", "log.csv", "train.log", "best.pt", "last.pt"} <= {p.name for p in run.iterdir()}
    record = json.loads((run / "config.json").read_text())
    assert record["head"] == "softmax" and record["seed"] == 3 and record["model_type"] == model_type
    assert record["train"]["max_steps"] == 6 and record["git_hash"] and record["normaliser"]["sigma_J"] > 0
    rows = list(csv.DictReader((run / "log.csv").open()))
    assert tuple(rows[0]) == LOG_COLUMNS and [int(r["step"]) for r in rows] == [0, 3, 6]
    assert summary["final_step"] == 6
    ckpt = torch.load(run / "last.pt", weights_only=True)
    assert {"model_state", "model_cfg", "train_cfg", "head", "model_type", "normaliser"} <= set(ckpt)
    stepper = load_stepper(run / "best.pt", "cpu")
    assert stepper.name == "smoke"
    with pytest.raises(FileExistsError):
        script.main(argv)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("head", HEADS)
def test_overfit_eight_trajectories(tmp_path, head):
    """Full-size GPT overfits 8 trajectories to < 1% of the persistence one-step L1 in 3000 steps."""
    data = TrajectoryData.load(fake_split(tmp_path / "f.h5", n=8, seed=0), "cuda")
    set_global_seed(0)
    model = build_model("gpt", head, {}, compute_normaliser(data.n, data.v)).cuda()
    cfg = TrainConfig(batch_size=8, lr=3e-4, warmup_steps=100, max_steps=3000, eval_every=3000, p_parity=0.0)
    train(model, data, data, cfg, tmp_path)
    metrics = evaluate(model, data, batch_size=8)
    ratio = metrics["val_l1"] / metrics["persistence_l1"]
    print(
        f"{head}: one-step L1 {metrics['val_l1']:.3e}, persistence {metrics['persistence_l1']:.3e}, ratio {ratio:.4f}"
    )
    assert ratio < 0.01
    # the reported L1 is the mean over frames of sum |n_hat - n| dx
    best = load_stepper(tmp_path / "last.pt", "cuda").predict_teacher_forced(data.n[:, :-1], data.v)
    assert l1_per_frame(best, data.n[:, 1:]).mean().item() == pytest.approx(metrics["val_l1"], rel=1e-6)
