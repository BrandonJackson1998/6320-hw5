#!/usr/bin/env python3
"""Plot Assignment 5 learning curves from BGG history CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "plots"

RUN_PAIRS = (
    ("baseline_unweighted_l2", "intervention_balanced_class_weight", "Baseline vs balanced class weight"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BGG logistic learning curves.")
    parser.add_argument("--history-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def plot_metric(
    history_dir: Path,
    run_names: list[str],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    for run_name in run_names:
        path = history_dir / run_name / f"{run_name}_history.csv"
        frame = pd.read_csv(path)
        plt.plot(frame["epoch"], frame[metric], marker="o", linewidth=2, label=run_name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_names = [pair[0] for pair in RUN_PAIRS] + [pair[1] for pair in RUN_PAIRS]
    plot_metric(args.history_dir, run_names, "train_loss", "Training loss (log loss)", args.output_dir / "training_loss_curves.png")
    plot_metric(args.history_dir, run_names, "validation_loss", "Validation loss (log loss)", args.output_dir / "validation_loss_curves.png")
    plot_metric(args.history_dir, run_names, "validation_f1", "Validation F1 (high_rating)", args.output_dir / "validation_f1_curves.png")
    plot_metric(args.history_dir, run_names, "validation_recall", "Validation recall (high_rating)", args.output_dir / "validation_recall_curves.png")
    print(f"wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
