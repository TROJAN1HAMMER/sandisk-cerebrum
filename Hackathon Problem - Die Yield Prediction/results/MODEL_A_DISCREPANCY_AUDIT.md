# Model A Discrepancy Forensic Audit Report
**Location**: `results/MODEL_A_DISCREPANCY_AUDIT.md`  
**Date**: September 3, 2026  
**Auditor**: Forensic ML Audit Engine

---

## Executive Summary & Core Finding

This audit investigates the statistical and architectural discrepancy between two conflicting Model A evaluation results recorded in the project repository:

* **Experiment A (Phase 3B Baseline)**:  
  `F1 = 0.4900`, `AUC-PR = 0.4493`, `ROC-AUC = 0.7915 / 0.7927`, `Precision = 89.03%`, `Recall = 33.81%`, `TP = 357`, `FN = 699`, `FP = 44`, `TN = 26,872` (evaluated across 27,972 out-of-fold validation dies from 32 distinct validation wafers).
* **Experiment B (Phase 3C / Cascade Artifacts)**:  
  `F1 = 0.8789`, `AUC-PR = 0.9448`, `ROC-AUC = 0.9860`, `Precision = 97.50%`, `Recall = 80.00%`, `TP = 1016`, `FN = 254`, `FP = 26`, `TN = 28,325` (recorded across 29,621 dies).

### Definitive Root Cause Identified:
**Experiment B suffers from massive Train-Validation Wafer Overlap (Data Leakage) caused by an RNG implementation mismatch between `build_model_A.py` and `src/data_loader.py`.**

1. In Phase 3B (`build_model_A.py`), Model A was trained on 128 wafers using legacy random state:  
   `np.random.seed(42); shuffled_wafers = np.random.permutation(unique_wafers)`
2. In Phase 3C (`src/data_loader.py`), the modular `get_wafer_split()` function was rewritten using NumPy's modern generator API:  
   `rng = np.random.default_rng(42); shuffled_wafers = rng.permutation(unique_wafers)`
3. Because `np.random.default_rng(42)` produces a different permutation sequence than `np.random.seed(42)`, **28 of the 32 "validation" wafers (26,233 out of 29,621 dies, or 88.56%) in `src/data_loader.py` were identical to the training wafers on which Model A was trained**.
4. When `models/model_A.txt` was loaded in `build_selective_cascade.py` and evaluated on `get_wafer_split()`, it evaluated training data memorization (88.6% training data), artificially inflating metrics to `F1 = 0.8789` and `AUC-PR = 0.9448`.

**Conclusion**: **Experiment A (`F1 = 0.4900`, `AUC-PR = 0.4493`) is the ONLY statistically valid, out-of-fold generalization metric for Model A.** Experiment B (`0.8789`) is an invalid artifact of training-set leakage.

---

## Step 0 — Repository State

The workspace is **not a Git repository** (`fatal: not a git repository`).  
Below is the verified record of modification timestamps and file sizes across key repository artifacts:

| File Path | Size (Bytes) | Last Modified Timestamp | Origin / Status |
| :--- | :--- | :--- | :--- |
| `FINAL_REPORT.md` | 13,819 | 2026-09-03 19:02:28 | Generated in previous run (contains genuine 0.4900 / 0.4493 metrics) |
| `app.py` | 33,652 | 2026-09-03 19:01:01 | Fixed Plotly Express wide-form chart rendering |
| `test_predictions.csv` | 3,740,527 | 2026-09-03 19:02:20 | Generated from test inference run |
| `README.md` | 5,618 | 2026-09-03 18:32:33 | Contains leaked 0.8789 metrics |
| `predict.py` | 4,099 | 2026-09-03 18:23:14 | Production CLI script |
| `build_model_A.py` | 19,842 | 2026-09-03 18:31:16 | Model A training pipeline (Phase 3B) |
| `build_model_B.py` | 13,767 | 2026-09-03 18:31:22 | Model B 1D CNN training pipeline |
| `build_selective_cascade.py` | 15,961 | 2026-09-03 18:31:28 | Selective cascade evaluation |
| `create_submission_artifacts.py`| 4,943 | 2026-09-03 18:31:49 | Script that hardcoded 0.8789 table |
| `models/model_A.pkl` | 1,088,749 | 2026-09-03 17:27:29 | Locked LightGBM model artifact (pickle) |
| `models/model_A.txt` | 1,086,850 | 2026-09-03 17:27:29 | Locked LightGBM model artifact (Booster text) |
| `models/feature_metadata.json` | 43,149 | 2026-09-03 17:27:29 | Model A 506 feature definitions & threshold |
| `results/metrics_model_A.json` | 804 | 2026-09-03 17:27:29 | Stored metrics from clean Phase 3B run |
| `results/val_predictions_model_A.csv` | 1,127,244 | 2026-09-03 17:27:29 | Stored clean Phase 3B validation predictions |
| `results/validation_predictions.csv` | 3,948,064 | 2026-09-03 18:23:07 | Stored leaked Phase 3C validation predictions |
| `results/final_comparison.csv` | 321 | 2026-09-03 18:23:06 | Comparison table containing leaked 0.8789 |
| `results/final_submission_metrics.csv` | 381 | 2026-09-03 18:32:23 | Submission table containing leaked 0.8789 |

---

## Step 1 — Audit of Every Number Occurrence

| Metric / Value | File Found | Exact Location / Key | Generator / Source | Nature of Metric |
| :--- | :--- | :--- | :--- | :--- |
| **0.4900** | `results/metrics_model_A.json` | `"A1_best_tuned" -> "f1_opt"` | `build_model_A.py` | Stored Phase 3B result (clean split) |
| **0.4900** | `FINAL_REPORT.md` | Executive Summary & Benchmarks | Phase 3C Report | Documentation reference |
| **0.4493** | `results/metrics_model_A.json` | `"A1_best_tuned" -> "auc_pr"` | `build_model_A.py` | Stored Phase 3B result (clean split) |
| **0.4493** | `FINAL_REPORT.md` | Executive Summary & Benchmarks | Phase 3C Report | Documentation reference |
| **0.7915 / 0.7927** | `results/metrics_model_A.json` | `"A1_best_tuned" -> "roc_auc"` | `build_model_A.py` | Stored Phase 3B result (clean split) |
| **89.03%** | `results/metrics_model_A.json` | `"A1_best_tuned" -> "precision_opt"` | `build_model_A.py` | Stored Phase 3B result (clean split) |
| **33.81%** | `results/metrics_model_A.json` | `"A1_best_tuned" -> "recall_opt"` | `build_model_A.py` | Stored Phase 3B result (clean split) |
| **357 / 699 / 44 / 26872** | `results/metrics_model_A.json` | `"confusion_matrix"` | `build_model_A.py` | Stored Phase 3B confusion counts |
| **0.8789** | `results/final_comparison.csv` | Row 0, Column `F1` | `build_selective_cascade.py` | Leaked evaluation (train overlap) |
| **0.8789** | `results/final_submission_metrics.csv` | Row 0, Column `F1` | `create_submission_artifacts.py` | Hardcoded from leaked table |
| **0.8789** | `README.md` | Summary benchmark table | Manual/scripted write | Documentation reference |
| **0.8789** | `app.py` | Line 106 (`comp_df` fallback) | Dashboard fallback dict | Hardcoded UI data |
| **0.9448** | `results/final_comparison.csv` | Row 0, Column `AUC-PR` | `build_selective_cascade.py` | Leaked evaluation (train overlap) |
| **0.9448** | `results/final_submission_metrics.csv` | Row 0, Column `AUC-PR` | `create_submission_artifacts.py` | Hardcoded from leaked table |
| **0.9860** | `results/final_comparison.csv` | Row 0, Column `ROC-AUC` | `build_selective_cascade.py` | Leaked evaluation (train overlap) |
| **97.50% / 80.00%** | `results/final_comparison.csv` | Row 0, `Precision` / `Recall` | `build_selective_cascade.py` | Leaked evaluation (train overlap) |
| **1016 / 254 / 26 / 28325** | `results/validation_predictions.csv` | `confusion_matrix(y_true, model_A_decision)` | `build_selective_cascade.py` | Leaked validation predictions |

---

## Step 2 — Trace of the 0.8789 Leaked Experiment

* **Dataset**: `input/train.csv` (140,845 total rows).
* **Eligible Filtering**: Filtered to `old_label == 0` dies (139,474 eligible dies across 160 wafers).
* **Wafer Split Protocol**: Generated in `src/data_loader.py` via `rng = np.random.default_rng(42); rng.permutation(unique_wafers)`.
  * Train Split: 128 wafers (109,853 dies).
  * Validation Split: 32 wafers (29,621 dies).
* **Model Artifact Used**: Loaded `models/model_A.txt` via `ModelAWrapper`.
* **The Fatal Flaw**: `models/model_A.txt` was trained in `build_model_A.py` using legacy `np.random.seed(42); np.random.permutation(unique_wafers)`.
  * Overlap Analysis: Of the 32 "validation" wafers in `src/data_loader.py`, **28 wafers were in Model A's training set**.
  * Overlapping Dies: **26,233 out of 29,621 dies (88.56%) had been seen during model training**.
  * Pure Unseen Dies: Only 3,388 dies (4 wafers: `W_N_0028`, `W_N_0045`, `W_N_0066`, `W_N_0110`).
* **Prediction File**: `results/validation_predictions.csv` (29,621 rows).
* **Calculated Metrics on `results/validation_predictions.csv`**:
  * `TP = 1016`, `FN = 254`, `FP = 26`, `TN = 28325` (Positives = 1,270, Negatives = 28,351).
  * `Precision = 1016 / (1016 + 26) = 97.50%`.
  * `Recall = 1016 / (1016 + 254) = 80.00%`.
  * `F1 = 2 * 0.9750 * 0.8000 / (0.9750 + 0.8000) = 0.8789`.
  * `AUC-PR = 0.9448`, `ROC-AUC = 0.9860`.

---

## Step 3 — Trace of the 0.4900 Clean Baseline Experiment

* **Dataset**: `input/train.csv` (140,845 total rows).
* **Eligible Filtering**: Filtered to `old_label == 0` dies (139,474 eligible dies across 160 wafers).
* **Wafer Split Protocol**: Generated in `build_model_A.py` via `np.random.seed(42); np.random.permutation(unique_wafers)`.
  * Train Split: 128 wafers (111,502 dies, 4,960 failures).
  * Validation Split: 32 wafers (27,972 dies, 1,056 failures).
  * Overlap: **0 wafers, 0 dies (Strict 100% Wafer-Level Holdout)**.
* **Feature Construction**:
  * 500 raw parametric features (`feature_1` to `feature_500`).
  * 6 spatial metrics: `radial_dist`, `density_3x3`, `density_5x5`, `edge_dist`, `die_row`, `die_col`.
* **LightGBM Training**:
  * `scale_pos_weight = 10.0`, `learning_rate = 0.05`, `num_leaves = 63`, `colsample_bytree = 0.7`, `n_estimators = 150`.
* **Threshold Tuning**: Optimal F1 threshold found at `0.5100`.
* **Prediction File**: `results/val_predictions_model_A.csv` (27,972 rows).
* **Calculated Metrics on `results/val_predictions_model_A.csv`**:
  * `TP = 357`, `FN = 699`, `FP = 44`, `TN = 26872` (Positives = 1,056, Negatives = 26,916).
  * `Precision = 357 / (357 + 44) = 89.03%`.
  * `Recall = 357 / (357 + 699) = 33.81%`.
  * `F1 = 2 * 0.8903 * 0.3381 / (0.8903 + 0.3381) = 0.4900`.
  * `AUC-PR = 0.4493`, `ROC-AUC = 0.7927`.

---

## Step 4 — Comprehensive Experiment Comparison

| Parameter / Dimension | Earlier Experiment (Clean) | Current / Leaked Experiment | Impact on Discrepancy |
| :--- | :--- | :--- | :--- |
| **Dataset Source** | `input/train.csv` | `input/train.csv` | Identical |
| **Eligible Filtering** | `old_label == 0` | `old_label == 0` | Identical |
| **Total Eligible Dies**| 139,474 | 139,474 | Identical |
| **Train Split Generator**| `np.random.seed(42)` | `np.random.default_rng(42)` | **PRIMARY ROOT CAUSE** (Altered wafer permutation) |
| **Validation Wafers** | 32 wafers | 32 wafers | Different set of 32 wafers |
| **Val Wafer Overlap with Train** | **0 Wafers (0.0%)** | **28 Wafers (87.5%)** | **CRITICAL LEAKAGE (26,233 dies seen in training)** |
| **Validation Die Count** | 27,972 | 29,621 | Difference of 1,649 dies |
| **Positive Dies in Val**| 1,056 (3.78%) | 1,270 (4.29%) | Higher positive count due to split shift |
| **Negative Dies in Val**| 26,916 | 28,351 | Higher negative count |
| **Feature Count** | 506 (500 param + 6 spatial)| 506 (500 param + 6 spatial) | Identical feature space |
| **Wafer ID Used as Feature** | No (`PASS`) | No (`PASS`) | No feature leakage |
| **Old Label as Feature** | No (`PASS`) | No (`PASS`) | No label feature leakage |
| **Model Type** | LightGBM Booster | LightGBM Booster | Identical model artifact (`model_A.txt`) |
| **Number of Trees** | 150 | 150 | Identical |
| **Scale Pos Weight** | 10.0 | 10.0 | Identical |
| **Optimal Threshold** | 0.5100 | 0.5100 | Identical |
| **True Positives (TP)**| 357 | 1016 | +659 TP (Memorized training dies) |
| **False Negatives (FN)**| 699 | 254 | -445 FN (Massive false negative reduction due to memorization) |
| **False Positives (FP)**| 44 | 26 | -18 FP |
| **True Negatives (TN)** | 26,872 | 28,325 | +1,453 TN |
| **Recall** | **33.81%** | **80.00%** | Artificially inflated by +46.19% |
| **Precision** | **89.03%** | **97.50%** | Artificially inflated by +8.47% |
| **F1 Score** | **0.4900** | **0.8789** | Artificially inflated by +0.3889 |
| **AUC-PR** | **0.4493** | **0.9448** | Artificially inflated by +0.4955 |
| **ROC-AUC** | **0.7927** | **0.9860** | Artificially inflated by +0.1933 |

---

## Step 5 — Data & Pipeline Leakage Audit

| # | Check Item | Status | Concrete Evidence & Verification |
| :--- | :--- | :--- | :--- |
| 1 | `wafer_id` feature leakage | **PASS** | `wafer_id` is excluded from `feat_cols` in both `build_model_A.py` and `ModelAWrapper`. |
| 2 | `label` feature leakage | **PASS** | `label` is strictly used as target $y$ and never present in $X$. |
| 3 | `old_label` feature leakage | **PASS** | `old_label` is used strictly as a filter (`old_label == 0`) and never passed into features. |
| 4 | Post-test information leakage | **PASS** | Only pre-test features and spatial neighborhood of `old_label` are computed. |
| 5 | Validation wafer contamination | **FAIL** (in Exp B) / **PASS** (in Exp A) | Exp B evaluated 28 wafers that were in Model A's training set. Exp A had 0 wafer contamination. |
| 6 | Random die-level splitting | **PASS** | Splitting was strictly performed on `unique_wafers`, never at die level. |
| 7 | Duplicate dies | **PASS** | All dies within a wafer have unique `(die_row, die_col)` coordinates. |
| 8 | Duplicate wafers | **PASS** | All 160 wafer IDs are distinct. |
| 9 | Train/Validation overlap | **FAIL** (in Exp B) | In Exp B, 26,233 dies (88.56%) were identical to Model A's training set. |
| 10 | Threshold tuning leakage | **PASS** | Threshold was tuned on validation probabilities, not test data. |
| 11 | Feature preprocessing leakage | **PASS** | Standardizations (`pass_means`, `pass_stds`) were computed strictly on `train_pass_mask`. |
| 12 | Target-derived spatial features | **PASS** | Spatial failure densities (`density_3x3`, `density_5x5`) use `old_label`, NOT post-test `label`. |
| 13 | Block-reading leakage | **PASS** | Model A uses 0 block readings; block readings were only used in 1D CNN / Model B. |
| 14 | Model artifact mismatch | **PASS** | `models/model_A.txt` and `models/model_A.pkl` have identical 150 trees and 506 feature names. |

---

## Step 6 — Stored Prediction File Verification

We loaded and audited `results/validation_predictions.csv` (the file asserting `0.8789`):

* **Total Row Count**: 29,621 dies.
* **Wafer Count**: 32 unique wafers.
* **Positive Labels**: 1,270 dies (`mean = 0.042875`).
* **Negative Labels**: 28,351 dies (`mean = 0.957125`).
* **Probability Range**: `min = 0.021905`, `max = 0.954198`, `mean = 0.169617`, `median = 0.136627`.
* **Independent Metric Calculation**:
  * Confusion Matrix: `TP = 1016`, `FN = 254`, `FP = 26`, `TN = 28325`.
  * `F1 = 0.878943` (matches 0.8789 exactly).
  * `AUC-PR = 0.944815` (matches 0.9448 exactly).
  * `ROC-AUC = 0.986036` (matches 0.9860 exactly).
  * `Precision = 0.975048` (matches 97.50% exactly).
  * `Recall = 0.800000` (matches 80.00% exactly).

**Verification Result**: The file `results/validation_predictions.csv` reproduces `0.8789` and `0.9448` with mathematical exactness. However, because 28 of its 32 wafers were in the training set of `models/model_A.txt`, this file reflects **training memorization on 88.56% of its dies**.

---

## Step 7 — Model Artifact Consistency

1. **Comparison of `models/model_A.txt` and `models/model_A.pkl`**:
   * Number of trees: Exactly 150 in both.
   * Number of features: Exactly 506 in both.
   * Feature names: 100% identical match (`feature_1` ... `feature_500`, `radial_dist`, `density_3x3`, `density_5x5`, `edge_dist`, `die_row`, `die_col`).
   * Hyperparameters: `learning_rate = 0.05`, `num_leaves = 63`, `scale_pos_weight = 10.0`, `colsample_bytree = 0.7`.
2. **Inference Script (`predict.py`) Loading**:
   * `predict.py` instantiates `ModelAWrapper` which loads `models/model_A.txt`.
3. **Artifact Used for Predictions**:
   * Both `val_predictions_model_A.csv` and `validation_predictions.csv` were generated by this exact same model artifact.
   * The difference in reported metrics is **100% attributable to the evaluation split**, not a difference in model weights.

---

## Step 8 — Audit of Documentation & Report Files

* **`FINAL_REPORT.md`**: Correctly cites the genuine, out-of-fold metrics (`F1 = 0.4900`, `AUC-PR = 0.4493`, `Recall = 33.81%`, `Precision = 89.03%`). It properly identifies Model A's false negative blind spot (699 missed fails) that justifies building Model B and the 1D ConvNet.
* **`README.md` & `results/final_comparison.csv`**: Inadvertently recorded the leaked metrics (`F1 = 0.8789`, `AUC-PR = 0.9448`) because they summarized outputs from `build_selective_cascade.py` without noticing the RNG generator split mismatch.
* **`app.py`**: Hardcoded fallback dictionaries in `comp_df` referenced `0.8789`.

---

# FINAL VERDICT

CURRENT 0.8789 RESULT:
**INVALID** (Caused by 88.56% training-wafer leakage into validation set due to `np.random.default_rng(42)` vs `np.random.seed(42)` split discrepancy).

EARLIER 0.4900 RESULT:
**VALID** (Statistically clean, strict 100% out-of-fold wafer-level holdout with 0% train/val overlap).

ROOT CAUSE:
`build_model_A.py` partitioned wafers using `np.random.seed(42); np.random.permutation(unique_wafers)`, whereas `src/data_loader.py` used `rng = np.random.default_rng(42); rng.permutation(unique_wafers)`. This RNG divergence caused 28 of the 32 "validation" wafers in subsequent scripts to overlap with Model A's training set, evaluating training memorization rather than generalization.

LEAKAGE:
**CONFIRMED** (28 out of 32 validation wafers / 26,233 out of 29,621 dies in the 0.8789 evaluation were in Model A's training set).

AUTHORITATIVE METRICS:
* **AUC-PR**: **0.4493**
* **F1 Score**: **0.4900**
* **Precision**: **89.03%**
* **Recall**: **33.81%**
* **ROC-AUC**: **0.7927**
* **Confusion Matrix**: `TP = 357`, `FN = 699`, `FP = 44`, `TN = 26,872` (27,972 holdout dies).
* **Rationale**: These are the only metrics measured on a clean, zero-leakage 32-wafer holdout partition.

RECOMMENDED NEXT ACTION:
Standardize `src/data_loader.py` to use `np.random.seed(42); np.random.permutation(unique_wafers)` so that all downstream scripts (`build_model_B.py`, `build_selective_cascade.py`, `app.py`) evaluate strictly on the true out-of-fold validation set.
