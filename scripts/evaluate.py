"""Evaluate a model (or a trivial baseline) by free rollout on one or more data splits.

Examples
--------
    python scripts/evaluate.py --model persistence --splits test ood_field --out results/persistence
    python scripts/evaluate.py --model checkpoints/gpt_flux_s0/best.pt --splits test ood_field ood_Z ood_family

Writes ``{out}/{split}.h5`` per split and merges the per-split summaries into ``{out}/summary.json``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from lqm.eval.baselines import LinearExtrapolationStepper, PersistenceStepper
from lqm.eval.runner import evaluate

logger = logging.getLogger("evaluate")

BASELINES = {"persistence": PersistenceStepper, "linear": LinearExtrapolationStepper}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="'persistence', 'linear' or a checkpoint path (e.g. best.pt)")
    p.add_argument("--splits", nargs="+", default=["test"], help="split names, read from {data-dir}/{split}.h5")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=None, help="output directory (default results/<model name>)")
    p.add_argument("--n-steps", type=int, default=None, help="rollout length (default nt-1 = 200)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-traj", type=int, default=None, help="evaluate only the first N trajectories")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-save", type=int, default=32, help="store predicted densities of the first N trajectories")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    meta: dict[str, str] = {}
    if args.model in BASELINES:
        stepper = BASELINES[args.model]()
    else:
        from lqm.model.checkpoint import load_stepper  # lazy: only needed for trained models

        ckpt = Path(args.model)
        stepper = load_stepper(ckpt, args.device)
        meta["checkpoint"] = str(ckpt.resolve())
    out = args.out if args.out is not None else Path("results") / stepper.name

    for split in args.splits:
        summary = evaluate(
            stepper,
            args.data_dir / f"{split}.h5",
            out / f"{split}.h5",
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            device=args.device,
            max_traj=args.max_traj,
            seed=args.seed,
            n_save=args.n_save,
            meta=meta,
        )
        for tq, s in summary["rollout_l1"].items():
            logger.info(
                "%s t=%s: L1 mean %s [%s, %s], median %s", split, tq, s["mean"], s["ci_lo"], s["ci_hi"], s["median"]
            )
    logger.info("results in %s", out)


if __name__ == "__main__":
    main()
