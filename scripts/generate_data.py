#!/usr/bin/env python
"""Generate the exact LQM-1e dataset splits and write ``{out}/{split}.h5``.

The work is one 128x128 ``eigh`` pair plus a few small GEMMs per trajectory, so it does not
benefit from many BLAS threads: on a 16-thread i9-11900K, ``OPENBLAS_NUM_THREADS=4`` is ~1.3x
faster than the default for the propagation and ~9x faster for the ``ood_family`` potential
search, which calls LAPACK on a single small matrix at a time.

Examples
--------
Full dataset (~2.5 min, ~6.2 GB on disk)::

    OPENBLAS_NUM_THREADS=4 python scripts/generate_data.py --out data --base-seed 1234 \
        --n-train 24000 --n-val 3000 --n-test 3000 --n-ood 1000

A small smoke test of every split::

    python scripts/generate_data.py --out /tmp/lqm_small --n-train 200 --n-val 50 \
        --n-test 50 --n-ood 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from lqm import constants as C
from lqm.io import write_split
from lqm.physics.generate import default_config, generate_split

logger = logging.getLogger("generate_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    p.add_argument("--config", type=Path, default=Path("configs/data.json"), help="generation config JSON")
    p.add_argument("--base-seed", type=int, default=None, help="base seed (default: from config)")
    p.add_argument("--n-train", type=int, default=None, help="trajectories in train")
    p.add_argument("--n-val", type=int, default=None, help="trajectories in val")
    p.add_argument("--n-test", type=int, default=None, help="trajectories in test")
    p.add_argument("--n-ood", type=int, default=None, help="trajectories in each ood_* split")
    p.add_argument("--splits", nargs="+", default=list(C.SPLITS), choices=list(C.SPLITS), help="splits to generate")
    p.add_argument("--chunk-size", type=int, default=128, help="trajectories per batched propagation")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate and write the requested splits. Returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not any(os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")):
        logger.info("hint: OPENBLAS_NUM_THREADS=4 speeds this up (the matrices are only 128x128)")

    config = json.loads(args.config.read_text()) if args.config.exists() else default_config()
    if not args.config.exists():
        logger.warning("%s not found, using the built-in default config", args.config)
    if args.base_seed is not None:
        config["base_seed"] = args.base_seed
    for name, value in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        if value is not None:
            config["counts"][name] = value
    if args.n_ood is not None:
        for name in C.SPLITS:
            if name.startswith("ood_"):
                config["counts"][name] = args.n_ood

    base_seed = int(config["base_seed"])
    total = time.perf_counter()
    for split in args.splits:
        n = int(config["counts"][split])
        logger.info("generating %s: %d trajectories, base_seed %d", split, n, base_seed)
        start = time.perf_counter()
        arrays, attrs = generate_split(split, n, base_seed, chunk_size=args.chunk_size, config=config)
        path = args.out / f"{split}.h5"
        write_split(path, arrays, attrs)
        size_gb = path.stat().st_size / 1024**3
        logger.info(
            "%s: %d trajectories in %.1f s -> %s (%.2f GB)", split, n, time.perf_counter() - start, path, size_gb
        )
    logger.info("all requested splits done in %.1f s", time.perf_counter() - total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
