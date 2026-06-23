#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python3 scripts/prepare_bgg_data.py
python3 scripts/run_split_audit.py
python3 scripts/train_logistic_evaluation.py --epochs 40
python3 scripts/plot_learning_curves.py
python3 scripts/run_error_and_calibration.py \
  --predictions outputs/intervention_balanced_class_weight/validation_predictions.csv

echo "Local BGG evaluation complete. See prep/, outputs/, outputs/plots/"
