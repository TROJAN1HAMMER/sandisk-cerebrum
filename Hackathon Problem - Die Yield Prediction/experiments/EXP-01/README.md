# EXP-01: Spatial Defect Density Restoration & Baseline Retraining

This directory contains the complete artifact bundle for experiment `EXP-01`.

## Artifact Manifest

- `config.json`: Exact experiment configuration and hyperparameter specification.
- `feature_stats_before.csv`: Baseline spatial feature statistics (zeros).
- `feature_stats_after.csv`: Restored spatial feature statistics (non-zero distributions).
- `baseline_metrics.json`: Authoritative baseline metrics for Model A.
- `experiment_metrics.json`: Experimental metrics for Model A.
- `scorecard.csv`: Metric-by-metric comparison scorecard.
- `confusion_matrices.csv`: Confusion matrices for Model A and Cascade (baseline vs EXP-01).
- `wafer_metrics.csv`: Wafer-by-wafer breakdown across all 32 validation wafers.
- `feature_importance.csv`: Feature gain and split counts for all 506 features.
- `prediction_diagnostics.csv`: Die-by-die prediction comparisons across all 27,972 dies.
- `model_A_spatial.txt`: Isolated LightGBM booster model text artifact.
- `model_A_spatial.pkl`: Isolated LightGBM model pickle artifact.
- `hashes_before.json`: Pre-experiment SHA-256 hashes of production artifacts.
- `training_log.txt`: Full console output and execution log.
- `EXP-01_FINAL.md`: Full scientific evaluation report.
