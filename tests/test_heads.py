"""Head invariants: persistence at initialisation, exact norm conservation, flux consistency."""

import math

import pytest
import torch

from lqm import constants as C
from lqm.model.checkpoint import build_model
from lqm.model.heads import HEADS, Head
from lqm.model.normaliser import Normaliser

NM = Normaliser(sigma_n=0.04, sigma_v=0.3, sigma_delta=4e-4, sigma_J=6e-3)
D = 32
CONSERVING = ("unconstrained_meanfree", "softmax", "flux")


def _densities(B: int, T: int, seed: int) -> torch.Tensor:
    """Random float64 densities normalised in float64, with tails down to ~1e-14 (below N_FLOOR)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor(C.X)
    centre = 4 * torch.rand(B, T, 1, generator=g, dtype=torch.float64) - 2
    width = 0.5 + torch.rand(B, T, 1, generator=g, dtype=torch.float64)
    n = torch.exp(-(((x - centre) / width) ** 2)) + 1e-14
    return n / (n.sum(-1, keepdim=True) * C.DX)


def _randomise(head: Head, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        head.proj.weight.copy_(torch.randn(head.proj.weight.shape, generator=g) / math.sqrt(D))
        head.proj.bias.copy_(0.1 * torch.randn(head.proj.bias.shape, generator=g))


def _hidden(B: int, T: int, seed: int) -> torch.Tensor:
    return torch.randn(B, T, D, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("kind", HEADS)
def test_head_init_is_persistence(kind):
    n_k = _densities(3, 5, seed=0)
    n_next, J = Head(kind, D, NM)(_hidden(3, 5, 1), n_k)
    assert n_next.dtype == torch.float64 and n_next.shape == n_k.shape
    err = (n_next - n_k).abs().max().item()
    if kind == "softmax":
        assert err < 1e-6  # only the N_FLOOR clamp and renormalisation differ
    else:
        assert err == 0.0
    if kind == "flux":
        assert J.shape == (3, 5, C.NFACE) and J.dtype == torch.float64 and (J == 0).all()
    else:
        assert J is None


@pytest.mark.parametrize("kind", HEADS)
def test_model_init_is_persistence(kind):
    """The whole GPT / MLP is exact persistence at initialisation (float32 input, as in training)."""
    n = _densities(2, 6, seed=2).float()
    v = torch.randn(2, C.NX, dtype=torch.float64)
    for model_type in ("gpt", "mlp"):
        model = build_model(model_type, kind, {"n_layers": 2}, NM)
        err = (model(n, v)["n_next"] - n.double()).abs().max().item()
        assert err < 1e-6 if kind == "softmax" else err == 0.0, (model_type, err)


@pytest.mark.parametrize("kind", CONSERVING)
def test_head_conserves_norm_with_random_weights(kind):
    n_k = _densities(4, 7, seed=3)
    head = Head(kind, D, NM)
    _randomise(head, seed=4)
    n_next, _ = head(_hidden(4, 7, 5), n_k)
    assert (n_next - n_k).abs().max().item() > 1e-4  # the update is not trivially zero
    norm_change = ((n_next.sum(-1) - n_k.sum(-1)) * C.DX).abs().max().item()
    assert norm_change < 1e-13, norm_change


def test_unconstrained_head_does_not_conserve_norm():
    """Control: the norm test above is sensitive enough to see a non-conserving head."""
    n_k = _densities(4, 7, seed=3)
    head = Head("unconstrained", D, NM)
    _randomise(head, seed=4)
    n_next, _ = head(_hidden(4, 7, 5), n_k)
    assert ((n_next.sum(-1) - n_k.sum(-1)) * C.DX).abs().max().item() > 1e-6


def test_flux_head_update_matches_predicted_current():
    n_k = _densities(4, 7, seed=6)
    head = Head("flux", D, NM)
    _randomise(head, seed=7)
    n_next, J = head(_hidden(4, 7, 8), n_k)
    cum = torch.cumsum(n_next - n_k, dim=-1)
    J_rec = -(C.DX / C.DT) * cum[..., : C.NFACE]
    assert (J_rec - J).abs().max().item() < 1e-10
    assert cum[..., -1].abs().max().item() < 1e-14  # zero flux through the right wall


@pytest.mark.parametrize("kind", CONSERVING)
@pytest.mark.parametrize("model_type", ["gpt", "mlp"])
def test_model_conserves_norm_with_random_head(kind, model_type):
    """End to end with a float64 history, as in a rollout."""
    model = build_model(model_type, kind, {"n_layers": 2}, NM)
    _randomise(model.head, seed=9)
    n = _densities(2, 6, seed=10)
    v = torch.randn(2, C.NX, dtype=torch.float64)
    with torch.no_grad():
        n_next = model(n, v)["n_next"]
    assert (n_next - n).abs().max().item() > 1e-4
    assert ((n_next.sum(-1) - n.sum(-1)) * C.DX).abs().max().item() < 1e-13
