# CS 6320 — Assignment 5

**Name:** Brandon Jackson  
**Repo:** BGG portfolio baseline evaluation (Part A) and audit update (Part B).

## Contents

| Path | Purpose |
| --- | --- |
| `scripts/prepare_bgg_data.py` | Stratified train/validation/test split + manifest |
| `scripts/run_split_audit.py` | Split counts and distribution audits |
| `scripts/train_logistic_evaluation.py` | Baseline + intervention logistic runs with epoch history |
| `scripts/run_error_and_calibration.py` | Slices, confusion matrix, calibration bins |
| `scripts/plot_learning_curves.py` | Required learning-curve PNGs |
| `run_local.sh` | End-to-end local pipeline |
| `writeup/CS6320_Assignment5_Jackson.md` | Part A + Part B writeup |

## Data

Default source: `../6320-hw2/part_b/data/bgg/games.csv`

Override:

```bash
export GAMES_CSV=/path/to/games.csv
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run locally

```bash
bash run_local.sh
```

## Portfolio model (Assignment 4 charter)

- **Target:** `high_rating = 1` when `AvgRating >= 7.0`
- **Baseline:** logistic regression (SGD log-loss)
- **Intervention:** `class_weight=balanced` for minority-class recall
- **Split:** stratified 70/15/15 by game row, seed `6320`
