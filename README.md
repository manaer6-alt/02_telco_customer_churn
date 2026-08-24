**English** | [Русский](README_RU.md)

# Telco Customer Churn Prediction

An end-to-end customer churn project built around reproducible preprocessing and cost-aware threshold selection. The goal is not only to rank customers by churn risk, but to choose an intervention rule that can be connected to retention costs.

## Results

| Metric | Held-out test |
|---|---:|
| Accuracy | 0.7622 |
| Precision | 0.5399 |
| Recall | 0.7059 |
| F1 | 0.6118 |
| ROC-AUC | 0.8413 |
| PR-AUC | 0.6326 |

The selected balanced Logistic Regression uses a 0.60 threshold. On the held-out set it detected 264 of 374 churners, missed 110, and flagged 225 retained customers for review.

## What the project demonstrates

- Reproducible numeric and categorical preprocessing in a single scikit-learn pipeline.
- Honest model comparison on validation data.
- Threshold selection based on the relative cost of missed churn and unnecessary retention offers.
- Clear separation between ranking quality and the final business decision rule.

## Workflow

1. Validate types, missing values, and target balance.
2. Build a leakage-safe preprocessing pipeline.
3. Compare interpretable and non-linear classifiers.
4. Inspect ROC and precision-recall behavior.
5. Select a threshold and evaluate once on held-out data.

## Repository guide

- `notebooks/` — EDA, experiments, and final evaluation.
- `src/` — reusable code, where available.
- `reports/` — charts and project outputs.
- `requirements.txt` — recorded environment.

## Reproduce

Create a Python environment, install `requirements.txt`, and run the notebooks in order. The repository uses relative data paths.

## Limitations and next steps

The threshold should be recalculated from real retention economics and monitored as behavior changes. A stronger production study would add probability calibration, temporal validation, uplift modeling, and drift monitoring.

## Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn
