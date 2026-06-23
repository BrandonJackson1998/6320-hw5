#!/usr/bin/env python3
"""Error slices, confusion summaries, and calibration tables for BGG classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from bgg_common import DEFAULT_OUTPUT_DIR


def calibration_bins(frame: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    work = frame.copy()
    work["prob_bin"] = pd.qcut(work["confidence"], q=n_bins, duplicates="drop")
    grouped = (
        work.groupby("prob_bin", observed=False)
        .agg(
            count=("confidence", "size"),
            mean_predicted_prob=("confidence", "mean"),
            observed_positive_rate=("true_label", "mean"),
        )
        .reset_index()
    )
    grouped["calibration_gap"] = grouped["mean_predicted_prob"] - grouped["observed_positive_rate"]
    return grouped


def slice_report(frame: pd.DataFrame, slice_col: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(slice_col, observed=False):
        if len(group) < 20:
            continue
        report = classification_report(
            group["true_label"],
            group["predicted_label"],
            output_dict=True,
            zero_division=0,
        )
        rows.append(
            {
                "slice_column": slice_col,
                "slice_value": value,
                "count": int(len(group)),
                "positive_rate": float(group["true_label"].mean()),
                "accuracy": float(report["accuracy"]),
                "f1_macro": float(report["macro avg"]["f1-score"]),
                "f1_high_rating": float(report.get("1", {}).get("f1-score", 0.0)),
                "recall_high_rating": float(report.get("1", {}).get("recall", 0.0)),
                "precision_high_rating": float(report.get("1", {}).get("precision", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def binned_slice(frame: pd.DataFrame, column: str, bins: list[float], labels: list[str]) -> pd.DataFrame:
    work = frame.copy()
    work["slice"] = pd.cut(work[column], bins=bins, labels=labels, right=False)
    return slice_report(work, "slice")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BGG error, slice, and calibration analysis.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "intervention_balanced_class_weight" / "validation_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "error_analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(args.predictions)

    labels = sorted(set(preds["true_label"]) | set(preds["predicted_label"]))
    confusion = pd.DataFrame(
        confusion_matrix(preds["true_label"], preds["predicted_label"], labels=labels),
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    confusion.to_csv(args.output_dir / "confusion_matrix.csv")

    report = pd.DataFrame(
        classification_report(
            preds["true_label"],
            preds["predicted_label"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report.to_csv(args.output_dir / "classification_report.csv")

    calibration_bins(preds).to_csv(args.output_dir / "calibration_bins.csv", index=False)

    slice_report(preds, "YearPublished").to_csv(args.output_dir / "slice_by_year.csv", index=False)
    binned_slice(
        preds,
        "GameWeight",
        bins=[0, 2, 3, 4, float("inf")],
        labels=["light", "medium", "heavy", "very_heavy"],
    ).to_csv(args.output_dir / "slice_by_game_weight.csv", index=False)
    binned_slice(
        preds,
        "MinPlayers",
        bins=[0, 2, 3, float("inf")],
        labels=["solo_or_two", "three", "four_plus"],
    ).to_csv(args.output_dir / "slice_by_min_players.csv", index=False)

    high_conf_wrong = preds[(~preds["correct"]) & (preds["confidence"] >= 0.8)].sort_values(
        "confidence", ascending=False
    )
    high_conf_wrong.head(25).to_csv(args.output_dir / "high_confidence_errors.csv", index=False)

    print(f"Wrote error analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
