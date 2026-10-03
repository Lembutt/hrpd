"""Command-line interface for HRPD.

Example (full paper configuration, all five seeds):

.. code-block:: console

   hrpd --dataset cwru --data-dir data --output-dir results/cwru

Quick smoke run (a few minutes on GPU, useful to validate the setup):

.. code-block:: console

   hrpd --dataset mfpt --data-dir data --seeds 0 \
        --epochs-pretrain 5 --epochs-adapt 5 --output-dir /tmp/smoke

Outputs, per seed: ``probs_s{seed}.npy`` (test-set class probabilities)
and ``metrics_s{seed}.json``; plus an aggregate ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import DATASETS, HRPDConfig
from .data import load_base5
from .evaluate import compute_metrics, summarise_seeds
from .train import train_hrpd


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (separate for testability)."""
    p = argparse.ArgumentParser(
        prog="hrpd",
        description=(
            "Train and evaluate HRPD (Healthy-Referenced Physics-informed "
            "Diagnosis) on a packaged benchmark dataset."
        ),
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
        help="benchmark dataset to run on",
    )
    p.add_argument(
        "--data-dir",
        default="data",
        help="root directory containing <dataset>_base5_data folders",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="directory for probabilities and metric files (created if absent)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="generator/training seeds to run (default: all five)",
    )
    d = HRPDConfig()
    p.add_argument("--epochs-pretrain", type=int, default=d.epochs_pretrain,
                   help="stage-1 source pre-training epochs")
    p.add_argument("--epochs-adapt", type=int, default=d.epochs_adapt,
                   help="stage-2 adaptation epochs")
    p.add_argument("--batch-size", type=int, default=d.batch_size,
                   help="minibatch size per domain")
    p.add_argument("--lr", type=float, default=d.lr, help="AdamW learning rate")
    p.add_argument("--width", type=float, default=d.width,
                   help="physics-window half-width w")
    p.add_argument("--supcon-weight", type=float, default=d.supcon_weight,
                   help="weight of the supervised-contrastive loss")
    p.add_argument("--lambda-adv", type=float, default=d.lambda_adv,
                   help="weight of the adversarial loss")
    p.add_argument("--grl-coeff", type=float, default=d.grl_coeff,
                   help="fixed gradient-reversal coefficient")
    p.add_argument("--mixup-alpha", type=float, default=d.mixup_alpha,
                   help="Beta(alpha, alpha) parameter of Mixup")
    p.add_argument("--device", default=d.device,
                   help="torch device (cuda falls back to cpu if unavailable)")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    cfg = HRPDConfig(
        epochs_pretrain=args.epochs_pretrain,
        epochs_adapt=args.epochs_adapt,
        batch_size=args.batch_size,
        lr=args.lr,
        width=args.width,
        supcon_weight=args.supcon_weight,
        lambda_adv=args.lambda_adv,
        grl_coeff=args.grl_coeff,
        mixup_alpha=args.mixup_alpha,
        device=args.device,
    )
    spec = DATASETS[args.dataset]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in args.seeds:
        data = load_base5(args.data_dir, args.dataset, seed)
        probs, _ = train_hrpd(data, spec, cfg, seed)
        np.save(out / f"probs_s{seed}.npy", probs)
        metrics = compute_metrics(probs, data.yte, spec.n_classes)
        (out / f"metrics_s{seed}.json").write_text(json.dumps(metrics, indent=2))
        per_class = " ".join(
            f"{name}={rec:.2f}"
            for name, rec in zip(spec.class_names, metrics["per_class_recall"])
        )
        print(
            f"[{args.dataset}] seed {seed}: "
            f"balanced acc {metrics['balanced_accuracy']:.4f} | {per_class}",
            flush=True,
        )
        per_seed.append(metrics)

    summary = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "config": vars(args),
        **summarise_seeds(per_seed),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[{args.dataset}] {len(args.seeds)}-seed balanced acc "
        f"{summary['balanced_accuracy_mean']:.4f} "
        f"+/- {summary['balanced_accuracy_std']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
