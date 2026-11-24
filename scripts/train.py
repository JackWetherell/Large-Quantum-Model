"""Train one LQM-1e model (GPT or MLP baseline, any head).

Example
-------
    python scripts/train.py --config configs/train_gpt.json --head flux --seed 0 --data-dir data \
        --run-name gpt_flux_s0 [--window 2] [--model mlp] [--max-steps 20000]

Command-line options override the JSON config. Writes ``{runs_dir}/{run_name}/``: ``config.json``,
``log.csv``, ``train.log``, ``best.pt``, ``last.pt``.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from lqm.model.checkpoint import MODEL_TYPES, build_model, count_parameters, load_model
from lqm.model.heads import HEADS
from lqm.model.normaliser import compute_normaliser
from lqm.train.data import TrajectoryData
from lqm.train.loop import TrainConfig, train
from lqm.utils import git_hash, set_global_seed

logger = logging.getLogger("lqm.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="JSON config (configs/train_gpt.json, ...)")
    p.add_argument("--head", choices=HEADS, help="override the config's head")
    p.add_argument("--seed", type=int, help="override the config's seed")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="directory with train.h5 and val.h5")
    p.add_argument("--run-name", help="default: {model}_{head}[_w{W}]_s{seed}")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--model", choices=MODEL_TYPES, help="override the config's model_type")
    p.add_argument("--window", type=int, help="GPT attention window W")
    p.add_argument("--max-steps", type=int, help="override train.max_steps")
    p.add_argument("--pushforward-frac", type=float, help="override train.pushforward_frac")
    p.add_argument("--init-from", type=Path, help="checkpoint to fine-tune from (weights and normaliser)")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load the JSON config and apply command-line overrides."""
    cfg = json.loads(args.config.read_text())
    if args.model is not None:
        cfg["model_type"] = args.model
    if args.head is not None:
        cfg["head"] = args.head
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.window is not None:
        if cfg["model_type"] != "gpt":
            raise ValueError("--window only applies to the GPT")
        cfg["model"]["window"] = args.window
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.pushforward_frac is not None:
        cfg["train"]["pushforward_frac"] = args.pushforward_frac
    if cfg["model_type"] not in MODEL_TYPES or cfg["head"] not in HEADS:
        raise ValueError(f"bad model_type/head: {cfg['model_type']}/{cfg['head']}")
    return cfg


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    cfg = resolve_config(args)
    train_cfg = TrainConfig.from_dict(cfg["train"])
    window = cfg["model"].get("window")
    run_name = args.run_name or f"{cfg['model_type']}_{cfg['head']}{f'_w{window}' if window else ''}_s{cfg['seed']}"
    run_dir = args.runs_dir / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} exists and is not empty; choose another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")],
        force=True,
    )
    set_global_seed(cfg["seed"])
    device = torch.device(args.device)

    paths = {split: (args.data_dir / f"{split}.h5").resolve() for split in ("train", "val")}
    train_data = TrajectoryData.load(paths["train"], device)
    val_data = TrajectoryData.load(paths["val"], device)
    if args.init_from is not None:
        model, _ = load_model(args.init_from, device)
        if model.model_type != cfg["model_type"] or model.head.kind != cfg["head"]:
            raise ValueError("--init-from checkpoint does not match model_type/head")
        normaliser = model.normaliser
        logger.info("fine-tuning from %s (its normaliser is kept)", args.init_from)
    else:
        normaliser = compute_normaliser(train_data.n, train_data.v)
        model = build_model(cfg["model_type"], cfg["head"], cfg["model"], normaliser).to(device)

    record = {
        **cfg,
        "model": asdict(model.cfg),
        "data": {k: str(v) for k, v in paths.items()},
        "init_from": str(args.init_from) if args.init_from else None,
        "normaliser": normaliser.to_dict(),
        "n_params": count_parameters(model),
        "git_hash": git_hash(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "argv": sys.argv[1:] if argv is None else argv,
    }
    (run_dir / "config.json").write_text(json.dumps(record, indent=2))
    logger.info("run %s: %s %s, %d parameters", run_dir, cfg["model_type"], cfg["head"], record["n_params"])
    return train(model, train_data, val_data, train_cfg, run_dir)


if __name__ == "__main__":
    main()
