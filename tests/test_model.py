"""Backbone tests: parameter counts, causality, attention window, attention-weights path, stepper."""

import math

import pytest
import torch

from lqm import constants as C
from lqm.model.checkpoint import build_model, count_parameters, load_stepper, save_checkpoint
from lqm.model.normaliser import Normaliser
from lqm.model.stepper import Stepper

NM = Normaliser(sigma_n=0.04, sigma_v=0.3, sigma_delta=4e-4, sigma_J=6e-3)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _data(B: int, T: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Smoothly moving float64 densities (normalised) and a random potential."""
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor(C.X)
    t = C.DT * torch.arange(T, dtype=torch.float64)
    phase = 2 * math.pi * torch.rand(B, 1, 1, generator=g, dtype=torch.float64)
    centre = 2 * torch.sin(0.3 * t[None, :, None] + phase)
    n = torch.exp(-((x - centre) ** 2)) + 1e-12
    n = n / (n.sum(-1, keepdim=True) * C.DX)
    v = torch.randn(B, C.NX, generator=g, dtype=torch.float64)
    return n, v


def _model(model_type: str, head: str, device: str, window: int | None = None, seed: int = 0):
    """Full-size model with a random (non-zero) head so outputs depend on the backbone.

    The head scale gives per-frame updates max|n_hat - n_k| of order 1e-3..1e-2 for every head
    (already above realistic per-frame changes). The softmax head's delta is an unscaled log-ratio,
    so it needs a ~25x smaller weight scale than the sigma-scaled heads to stay in that range; at
    std 0.05 it would change n by >100% per frame, and float32 backbone noise (~1e-6 relative to
    the update, for every head) would then exceed the 1e-6 absolute stepper tolerance.
    """
    torch.manual_seed(seed)
    cfg = {"window": window} if model_type == "gpt" else {}
    model = build_model(model_type, head, cfg, NM).to(device).eval()
    std = 0.002 if head == "softmax" else 0.05
    with torch.no_grad():
        model.head.proj.weight.normal_(std=std)
        model.head.proj.bias.normal_(std=std)
    return model


def test_parameter_counts():
    counts = {
        (mt, h): count_parameters(build_model(mt, h, {}, NM))
        for mt in ("gpt", "mlp")
        for h in ("unconstrained", "flux")
    }
    print(counts)
    for c in counts.values():
        assert 3_000_000 <= c <= 5_000_000
    assert counts[("gpt", "unconstrained")] == 4_888_960
    assert counts[("mlp", "unconstrained")] == 4_846_424


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model_type", ["gpt", "mlp"])
def test_forward_outputs(model_type, device):
    model = _model(model_type, "flux", device)
    n, v = _data(2, 9, seed=0)
    out = model(n.float().to(device), v.to(device), return_attn=True)
    assert out["n_next"].shape == (2, 9, C.NX) and out["n_next"].dtype == torch.float64
    assert out["J"].shape == (2, 9, C.NFACE) and out["J"].dtype == torch.float64
    assert len(out["resid"]) == 7 and all(r.shape == (2, 9, model.cfg.d_model) for r in out["resid"])
    if model_type == "gpt":
        assert len(out["attn"]) == 6 and all(a.shape == (2, 8, 9, 9) for a in out["attn"])
    else:
        assert out["attn"] is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model_type,window", [("gpt", None), ("gpt", 2), ("gpt", 5), ("mlp", None)])
def test_causality(model_type, window, device):
    """Perturbing frame k+1 leaves every output at positions <= k bit-for-bit unchanged."""
    model = _model(model_type, "unconstrained", device, window)
    n, v = _data(2, 24, seed=1)
    n, v = n.to(device), v.to(device)
    k = 13
    n2 = n.clone()
    n2[:, k + 1] *= 1.0 + 0.1 * torch.rand(C.NX, dtype=torch.float64, device=device)
    with torch.no_grad():
        a, b = model(n, v), model(n2, v)
    assert torch.equal(a["n_next"][:, : k + 1], b["n_next"][:, : k + 1])
    for ra, rb in zip(a["resid"], b["resid"], strict=True):
        assert torch.equal(ra[:, : k + 1], rb[:, : k + 1])
    assert not torch.equal(a["n_next"][:, k + 1], b["n_next"][:, k + 1])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("window", [2, 5])
def test_window(window, device):
    """With window W the output at k depends on frames k-W+1..k only (also through deep layers)."""
    model = _model("gpt", "unconstrained", device, window)
    n, v = _data(2, 24, seed=2)
    n, v = n.to(device), v.to(device)
    k = 15
    with torch.no_grad():
        ref = model(n, v)["n_next"][:, k]
        n_old = n.clone()
        n_old[:, : k - window + 1] *= 1.1  # every frame before the window, incl. k-W
        n_in = n.clone()
        n_in[:, k - window + 1] *= 1.1  # the oldest frame inside the window
        assert torch.equal(model(n_old, v)["n_next"][:, k], ref)
        assert not torch.equal(model(n_in, v)["n_next"][:, k], ref)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("window", [None, 2, 5])
def test_attention_weights_path_matches_fast_path(window, device):
    model = _model("gpt", "flux", device, window)
    n, v = _data(3, 20, seed=3)
    n, v = n.to(device), v.to(device)
    with torch.no_grad():
        fast, slow = model(n, v), model(n, v, return_attn=True)
    assert fast["attn"] is None
    assert (fast["n_next"] - slow["n_next"]).abs().max().item() <= 1e-5
    for rf, rs in zip(fast["resid"], slow["resid"], strict=True):
        assert (rf - rs).abs().max().item() <= 1e-5
    T = n.shape[1]
    q = torch.arange(T, device=device)
    allowed = q[None, :] <= q[:, None]
    if window is not None:
        allowed &= q[:, None] - q[None, :] < window
    for a in slow["attn"]:
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-6)  # rows are distributions
        assert (a[..., ~allowed] == 0).all()  # nothing outside the causal window


def _save(model, path):
    save_checkpoint(path, model, train_cfg={"note": "test"}, step=0, val_loss=float("nan"))
    return path


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "model_type,window,head", [("gpt", None, "flux"), ("gpt", 2, "softmax"), ("mlp", None, "unconstrained")]
)
def test_stepper_consistency(model_type, window, head, device, tmp_path):
    """predict_next on n[:, :k+1] equals predict_teacher_forced(n)[:, k], and the checkpoint round-trips."""
    model = _model(model_type, head, device, window)
    run_dir = tmp_path / f"{model_type}_{head}"
    run_dir.mkdir()
    stepper = load_stepper(_save(model, run_dir / "best.pt"), device)
    assert isinstance(stepper, Stepper)
    assert stepper.name == run_dir.name
    assert stepper.context == {("gpt", None): 200, ("gpt", 2): 2, ("mlp", None): 2}[(model_type, window)]
    n, v = _data(3, 40, seed=4)  # float64 on the CPU: the stepper moves data to the model's device
    tf = stepper.predict_teacher_forced(n, v)
    assert tf.dtype == torch.float64 and tf.device == n.device and tf.shape == n.shape
    with torch.no_grad():
        direct = model(n.to(device), v.to(device))["n_next"].cpu()
    assert (tf - direct).abs().max().item() == 0.0  # checkpoint round trip is exact
    for k in (0, 1, 2, 7, 39):
        nxt = stepper.predict_next(n[:, : k + 1], v)
        assert nxt.shape == (3, C.NX) and nxt.dtype == torch.float64
        assert (nxt - tf[:, k]).abs().max().item() <= 1e-6, k


def test_stepper_teacher_forced_beyond_max_len(tmp_path):
    """A full 201-frame trajectory: the last position falls back to predict_next on the last 200 frames."""
    model = _model("gpt", "flux", "cpu")
    stepper = load_stepper(_save(model, tmp_path / "best.pt"), "cpu")
    n, v = _data(1, C.NT, seed=5)
    tf = stepper.predict_teacher_forced(n, v)
    assert tf.shape == (1, C.NT, C.NX)
    assert torch.equal(tf[:, :200], stepper.predict_teacher_forced(n[:, :200], v))
    assert torch.isfinite(tf).all()
