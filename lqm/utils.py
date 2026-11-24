"""Small shared helpers: reproducible seeding and provenance (frozen interface)."""

from __future__ import annotations

import logging
import random
import subprocess
from pathlib import Path

import numpy as np

from lqm import constants as C

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def trajectory_rng(base_seed: int, split: str, index: int) -> np.random.Generator:
    """Independent, reproducible RNG for trajectory ``index`` of ``split``.

    Uses ``SeedSequence([base_seed, split_id, index])`` so different splits can never share a
    stream, and any trajectory can be regenerated on its own.
    """
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), C.SPLIT_IDS[split], int(index)]))


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs, including CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def git_hash() -> str:
    """Current git commit hash, with a ``-dirty`` suffix if the tree has uncommitted changes."""
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        d = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return h.stdout.strip() + ("-dirty" if d.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        logger.warning("could not determine git hash")
        return "unknown"
