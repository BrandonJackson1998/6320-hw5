#!/usr/bin/env python3
"""Plot Assignment 5 learning curves from BGG history CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "plots"

RUNS: tuple[tuple[str, str, str], ...] = (
    ("baseline_unweighted_l2", "Baseline", "#1f77b4"),
    ("intervention_balanced_class_weight", "Intervention (balanced weights)", "#d62728"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BGG logistic learning curves.")
    parser.add_argument("--history-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_history(history_dir: Path, run_name: str) -> pd.DataFrame:
    path = history_dir / run_name / f"{run_name}_history.csv"
    return pd.read_csv(path)


def metric_delta(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].iloc[-1] - frame[column].iloc[0])


def plot_run_comparison(
    histories: dict[str, pd.DataFrame],
    val_metric: str,
    ylabel: str,
    output_path: Path,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Baseline vs intervention on validation — the comparison that actually moves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for run_name, label, color in RUNS:
        frame = histories[run_name]
        delta = metric_delta(frame, val_metric)
        ax.plot(
            frame["epoch"],
            frame[val_metric],
            linewidth=2.5,
            color=color,
            label=f"{label} (Δ={delta:+.4f} over 40 epochs)",
        )
        ax.scatter(
            [1, 40],
            [frame[val_metric].iloc[0], frame[val_metric].iloc[-1]],
            color=color,
            s=45,
            zorder=3,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Validation {ylabel} — baseline vs intervention")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small", loc="best")
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_generalization_check(histories: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Train vs validation gap — shows no divergence even though within-run curves are flat."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    panels = (
        ("train_loss", "validation_loss", "Log loss"),
        ("train_f1", "validation_f1", "F1"),
        ("train_recall", "validation_recall", "Recall"),
        ("train_roc_auc", "validation_roc_auc", "ROC-AUC"),
    )

    for ax, (train_col, val_col, title) in zip(axes.flat, panels):
        for run_name, label, color in RUNS:
            frame = histories[run_name]
            gap = frame[val_col] - frame[train_col]
            ax.plot(frame["epoch"], gap, linewidth=2, color=color, label=label)
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(f"Val − train: {title}")
        ax.set_ylabel("Gap")
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    axes[0, 0].legend(fontsize="small", loc="best")
    fig.suptitle("Generalization check (near-zero gap = train and validation track together)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_early_epochs(histories: dict[str, pd.DataFrame], output_path: Path) -> None:
    """First 5 epochs only — where the tiny within-run movement actually happens."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for run_name, label, color in RUNS:
        frame = histories[run_name].head(5)
        ax.plot(
            frame["epoch"],
            frame["validation_f1"],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1 (high_rating)")
    ax.set_title("Early convergence — validation F1, epochs 1–5")
    ax.set_xticks(range(1, 6))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    histories = {run_name: load_history(args.history_dir, run_name) for run_name, _, _ in RUNS}

    plot_run_comparison(
        histories,
        "validation_loss",
        "log loss",
        args.output_dir / "training_loss_curves.png",
    )
    plot_run_comparison(
        histories,
        "validation_f1",
        "F1 (high_rating)",
        args.output_dir / "validation_f1_curves.png",
        ylim=(0.50, 0.65),
    )
    plot_run_comparison(
        histories,
        "validation_recall",
        "recall (high_rating)",
        args.output_dir / "validation_recall_curves.png",
        ylim=(0.40, 0.75),
    )
    plot_run_comparison(
        histories,
        "validation_roc_auc",
        "ROC-AUC",
        args.output_dir / "validation_roc_auc_curves.png",
        ylim=(0.78, 0.85),
    )
    plot_generalization_check(histories, args.output_dir / "generalization_gap_curves.png")
    plot_early_epochs(histories, args.output_dir / "early_convergence_f1.png")
    print(f"wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
