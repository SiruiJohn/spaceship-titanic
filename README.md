# Spaceship Titanic Kaggle Project

This repository contains a Kaggle competition workflow for the
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
binary classification task.

## Project Overview

The goal is to predict whether each passenger was transported to another
dimension. The current pipeline covers:

- exploratory data analysis and visualization
- missing value imputation based on group, cabin, home planet, and spend logic
- feature engineering for passenger groups, cabins, surnames, spending behavior, and demographics
- OOF target encoding and ordinal encoding
- tree model ensembles with XGBoost, LightGBM, CatBoost, ExtraTrees, and HistGradientBoosting when available
- pseudo-label, MLP blending, SHAP, and calibration experiments
- final Kaggle submission generation

## Main Files

| Path | Purpose |
|---|---|
| `spaceship-titanic.py` | Main exported pipeline script |
| `spaceship-titanic.ipynb` | Main notebook version |
| `space-titanic-eda-advanced-feature-engineering.py` | Advanced EDA and feature engineering script |
| `eda_visualizations.py` | EDA visualization generator |
| `requirements.txt` | Python dependencies |
| `docs/progress/项目结构与提分路线.md` | Project structure and improvement roadmap |
| `EDA可视化分析报告.md` | EDA report |
| `模型训练与评估分析报告.md` | Model training and evaluation report |
| `spaceship-titanic-说明文档.md` | Technical pipeline documentation |

## Data

Kaggle data is expected locally at:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

Raw and processed data are ignored by Git to avoid committing generated or
competition data files.

## Current Reported Result

The project documentation records:

- OOF CV accuracy: about `81.87%`
- Kaggle public leaderboard score: `0.80453`
- main final submission: `submission_90plus.csv`

Generated submissions are ignored by Git; regenerate them by running the main
pipeline.

## Reproducibility Notes

Install dependencies first:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python spaceship-titanic.py
```

The current script treats `optuna` and `shap` as optional. If `optuna` is not
available, it falls back to default LightGBM parameters. If `shap` is not
available, SHAP analysis is skipped.

## Results Kept In Git

Only compact report assets are intended to be tracked:

- `results/eda_viz/*.png`
- `results/model_report/*.png`
- `results/metrics/**/*.txt`
- `results/params/*.txt`

Large notebook exports, logs, models, predictions, raw data, PDFs, and
submission CSVs stay local.

