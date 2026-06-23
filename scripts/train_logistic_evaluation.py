#!/usr/bin/env python3
"""Train BGG logistic baseline with epoch history and one generalization intervention."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from bgg_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREP_DIR,
    RANDOM_STATE,
    SPLIT_COL,
    TARGET_COL,
    build_feature_matrix,
    load_prepared_split,
    split_frame,
    write_json,
)


@dataclass(frozen=True)
class RunConfig:
    name: str
    description: str
    alpha: float
    class_weight: str | None


RUNS: list[RunConfig] = [
    RunConfig(
        name="baseline_unweighted_l2",
        description="Baseline logistic regression (SGD log-loss) with default L2 strength and no class weighting.",
        alpha=1e-4,
        class_weight=None,
    ),
    RunConfig(
        name="intervention_balanced_class_weight",
        description="Generalization intervention: class_weight=balanced to address minority-class recall on high_rating.",
        alpha=1e-4,
        class_weight="balanced",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BGG logistic models for Assignment 5.")
    parser.add_argument("--prep-dir", type=Path, default=DEFAULT_PREP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


def majority_metrics(y_true: np.ndarray) -> dict[str, float]:
    majority = 1 if y_true.mean() >= 0.5 else 0
    preds = np.full_like(y_true, fill_value=majority)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": 0.5,
    }


def fit_with_history(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: RunConfig,
    *,
    epochs: int,
    learning_rate: float,
) -> tuple[SGDClassifier, StandardScaler, pd.DataFrame]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=config.alpha,
        class_weight=None,
        learning_rate="constant",
        eta0=learning_rate,
        random_state=RANDOM_STATE,
        max_iter=1,
        warm_start=True,
    )

    history_rows = []
    classes = np.array([0, 1], dtype=int)
    if config.class_weight == "balanced":
        class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
        weight_map = {label: weight for label, weight in zip(classes, class_weights)}
        sample_weight = np.array([weight_map[label] for label in y_train])
    else:
        sample_weight = None

    for epoch in range(1, epochs + 1):
        model.partial_fit(
            x_train_scaled,
            y_train,
            classes=classes,
            sample_weight=sample_weight,
        )
        train_proba = model.predict_proba(x_train_scaled)[:, 1]
        val_proba = model.predict_proba(x_val_scaled)[:, 1]
        train_pred = (train_proba >= 0.5).astype(int)
        val_pred = (val_proba >= 0.5).astype(int)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(log_loss(y_train, train_proba, labels=[0, 1])),
                "validation_loss": float(log_loss(y_val, val_proba, labels=[0, 1])),
                "train_f1": float(f1_score(y_train, train_pred, zero_division=0)),
                "validation_f1": float(f1_score(y_val, val_pred, zero_division=0)),
                "train_roc_auc": float(roc_auc_score(y_train, train_proba)),
                "validation_roc_auc": float(roc_auc_score(y_val, val_proba)),
                "train_accuracy": float(accuracy_score(y_train, train_pred)),
                "validation_accuracy": float(accuracy_score(y_val, val_pred)),
                "train_precision": float(precision_score(y_train, train_pred, zero_division=0)),
                "validation_precision": float(precision_score(y_val, val_pred, zero_division=0)),
                "train_recall": float(recall_score(y_train, train_pred, zero_division=0)),
                "validation_recall": float(recall_score(y_val, val_pred, zero_division=0)),
            }
        )
    return model, scaler, pd.DataFrame(history_rows)


def save_predictions(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
) -> None:
    preds = (y_proba >= 0.5).astype(int)
    out = frame[["BGGId", "Name", "YearPublished", "GameWeight", "MinPlayers", "MaxPlayers"]].copy()
    out["true_label"] = y_true
    out["predicted_label"] = preds
    out["confidence"] = y_proba
    out["correct"] = out["true_label"] == out["predicted_label"]
    out.to_csv(output_path, index=False)


def run_one(
    prepared: pd.DataFrame,
    config: RunConfig,
    *,
    epochs: int,
    learning_rate: float,
    output_dir: Path,
) -> None:
    train = split_frame(prepared, "train")
    val = split_frame(prepared, "validation")
    test = split_frame(prepared, "test")

    x_train, y_train = build_feature_matrix(train)
    x_val, y_val = build_feature_matrix(val)
    x_test, y_test = build_feature_matrix(test)

    model, scaler, history = fit_with_history(
        x_train.to_numpy(),
        y_train.to_numpy(),
        x_val.to_numpy(),
        y_val.to_numpy(),
        config,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    run_dir = output_dir / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(run_dir / f"{config.name}_history.csv", index=False)

    x_val_scaled = scaler.transform(x_val.to_numpy())
    x_test_scaled = scaler.transform(x_test.to_numpy())
    val_proba = model.predict_proba(x_val_scaled)[:, 1]
    test_proba = model.predict_proba(x_test_scaled)[:, 1]

    save_predictions(val, y_val.to_numpy(), val_proba, run_dir / "validation_predictions.csv")
    save_predictions(test, y_test.to_numpy(), test_proba, run_dir / "test_predictions.csv")

    final_epoch = history.iloc[-1].to_dict()
    summary = {
        "experiment": config.name,
        "description": config.description,
        "model": "SGDClassifier(log_loss, l2)",
        "alpha": config.alpha,
        "class_weight": config.class_weight,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": RANDOM_STATE,
        "majority_baseline_validation": majority_metrics(y_val.to_numpy()),
        "majority_baseline_test": majority_metrics(y_test.to_numpy()),
        "final_epoch_metrics": final_epoch,
        "history_csv": str(run_dir / f"{config.name}_history.csv"),
    }
    write_json(run_dir / f"{config.name}_summary.json", summary)
    print(f"=== {config.name} ===")
    print(config.description)
    print(
        f"final val F1={final_epoch['validation_f1']:.3f} "
        f"recall={final_epoch['validation_recall']:.3f} "
        f"ROC-AUC={final_epoch['validation_roc_auc']:.3f}"
    )


def main() -> None:
    args = parse_args()
    prepared = load_prepared_split(args.prep_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for config in RUNS:
        run_one(
            prepared,
            config,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
        )
    write_json(
        args.output_dir / "experiment_manifest.json",
        {
            "primary_baseline": RUNS[0].name,
            "primary_intervention": RUNS[1].name,
            "runs": [run.__dict__ for run in RUNS],
        },
    )
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
