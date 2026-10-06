# Dataset Forensics: Complete Pipeline & Data Integrity Audit

**Location:** `audit/model_forensics/dataset_forensics.md`  
**Date:** October 6, 2026  
**Auditor:** Antigravity Forensic ML Engine  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

This forensic audit evaluates the complete dataset pipeline for the SanDisk Cerebrum Die Yield Prediction platform across all input files, pre-test filters, split protocols, coordinate systems, and signal representations.

### Key Empirical Findings:
1. **Three Primary Input Files + One Derived Benchmark Cohort:**
   * `input/train.csv`: 3,638.7 MB, 173,099 rows, 506 columns, 160 unique wafers.
   * `input/validation.csv`: 842.0 MB, 39,351 rows, 505 columns, 40 unique wafers (competition submission format without labels).
   * `input/test.csv`: 842.1 MB, 39,351 rows, 506 columns, 40 unique wafers (ground-truth test holdout).
   * `results/val_die_diagnostics.csv`: 6.72 MB, 27,972 rows, 24 columns, 32 unique wafers (canonical out-of-fold validation set).
2. **Zero Wafer Contamination Across All Boundaries:**
   * $\text{Train Wafers (128)} \cap \text{Holdout Validation Wafers (32)} = \emptyset$ (0 overlapping wafers).
   * $\text{Train CSV (160 wafers)} \cap \text{Test CSV (40 wafers)} = \emptyset$ (0 overlapping wafers).
   * $\text{Holdout Validation Wafers (32)} \cap \text{Test CSV (40 wafers)} = \emptyset$ (0 overlapping wafers).
   * Wafer isolation is 100% verified and strictly preserved.
3. **Pre-Test Eligibility Filter Discrepancy Resolved:**
   * In `input/train.csv`, total rows = 173,099. Dies with `old_label == 1` (pre-test rejects) = 19,062 (11.01%).
   * Eligible dies with `old_label == 0` = 154,037 dies.
   * Across the 32 canonical validation holdout wafers, eligible dies = **27,972**.
   * Defect prevalence among eligible dies = **1,056 actual post-test failures (3.7752%)**, yielding **96.2248% clean die yield**.
4. **Dead Spatial Neighborhood Density Features Discovered:**
   * In `src/spatial.py`, `compute_spatial_features(eligible)` was executed on already-filtered `eligible` dies without passing the unfiltered wafer map (`full_df=df`).
   * Because `old_label` is identically 0 for all eligible dies, `density_3x3` and `density_5x5` collapsed to **identically 0.0 across all 27,972 validation dies and 126,065 training dies**.
   * These two spatial density features provide zero split value to Model A and Model B in the current baseline.

---

## 2. Dataset Inventory Table

| Dataset File | File Type | Size | Row Count | Column Count | Wafers | Pre-Test Eligible (`old_label==0`) | Pre-Test Rejects (`old_label==1`) | Post-Test Defects (`label==1`) | Defect Rate | Missing / Duplicates |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `input/train.csv` | Raw Super-Dataset | 3.64 GB | 173,099 | 506 | 160 | 154,037 (88.99%) | 19,062 (11.01%) | 6,519 (eligible) | 4.23% (eligible) | 0 / 0 |
| `input/validation.csv` | Competition Holdout (Unlabeled) | 842.0 MB | 39,351 | 505 | 40 | 35,016 (88.98%) | 4,335 (11.02%) | Hidden | N/A | 0 / 0 |
| `input/test.csv` | Ground-Truth Competition Test | 842.1 MB | 39,351 | 506 | 40 | 35,016 (88.98%) | 4,335 (11.02%) | 1,481 (eligible) | 4.23% (eligible) | 0 / 0 |
| `results/val_die_diagnostics.csv` | Authoritative Validation Benchmark | 6.72 MB | 27,972 | 24 | 32 | 27,972 (100.0%) | 0 (filtered) | 1,056 | 3.7752% | 0 / 0 |

---

## 3. Data Generation Physics & Semiconductor Context

Inspection of `generate_data.py` reveals the exact semiconductor defect synthesis physics:
1. **Wafer Map Topology:** Sourced from actual semiconductor fabrication wafer maps (WM-811K benchmark). Each wafer has realistic geometry with circular wafer boundaries, edge exclusion zones, and notch orientations.
2. **Pre-Test vs Post-Test Labels:**
   * `old_label`: Pre-assembly wafer probe electrical test. Identifies gross functional failures (11.01% of all fabricated dies). These dies are physically rejected at wafer saw and never packaged.
   * `label`: Post-packaging burn-in / reliability stress test. The prediction target is predicting which pre-test passing dies (`old_label == 0`) will subsequently experience latent post-test failures (`label == 1`).
3. **Defect Sub-Mechanisms:**
   * **Marginal Failures (65% of post-test defects):** In `generate_data.py` line 364, marginal defects are injected with only 5% to 25% of normal electrical parameter shift (`base += fail_shift * rng.uniform(0.05, 0.25)`). These dies appear parametrically normal in tabular parametric tests (`feature_1` to `feature_500`), creating the severe False Negative blind spot in Model A.
   * **Sub-Die Block Anomalies:** In `generate_data.py` lines 436–451, memory blocks receive anomalous localized shifts (clustering around a seed block with mean shift $+4.5$, baseline 100.0). These sparse, localized excursions in the 2,000 sequential readings provide the physical signal captured by Model B's `cluster_area` and `b_mean`.

---

## 4. Per-Wafer Class Distribution & Yield Heterogeneity

Across the 32 validation holdout wafers (27,972 dies), yield and defect rates exhibit significant semiconductor process variation:
* **Total Dies per Wafer:** Range: 457 to 1,289 dies (Mean: 874.1, Median: 881.5).
* **Defect Count per Wafer:** Range: 2 to 106 defects (Mean: 33.0, Median: 24.5).
* **Wafer Failure Rate:** Range: 0.22% to 11.23% (Mean: 3.78%, Std: 2.61%).
* **Wafer Die Yield:** Range: 88.77% to 99.78% (Mean: 96.22%).

### Wafer Failure Rate Quartiles:
* **Min Failure Wafer:** `W_N_0060` (2 failures out of 893 dies = 0.22% fail rate, 99.78% yield).
* **25th Percentile:** 1.84% fail rate.
* **Median:** 3.12% fail rate.
* **75th Percentile:** 5.15% fail rate.
* **Max Failure Wafer:** `W_N_0107` (106 failures out of 944 dies = 11.23% fail rate, 88.77% yield).

**Implication:** Wafer defect rates vary by over **50x** (from 0.22% to 11.23%). Global metric averaging masks severe wafer-to-wafer performance dispersion.

---

## 5. Signal Integrity & Representation

* **Representation:** Space-separated ASCII string of 2,000 float values formatted to 2 decimal places (e.g., `"101.24 99.82 104.15 ..."`).
* **Length Consistency:** Exactly 2,000 values for 100% of rows. Zero truncated or extended records.
* **Missing Readings:** Zero `NaN` or `null` values.
* **Memory Footprint:** In raw CSV format, `block_readings` occupies ~18 KB per die, contributing over 85% of the 3.6 GB `input/train.csv` file size.
* **Parsing Throughput:** Reading and parsing 2,000 block floats for 27,972 dies requires ~4.2 seconds in pure NumPy vectorized parsing (`src/block_parser.py:parse_block_matrix`).

---

## 6. Dead Feature Audit

| Feature | Feature Family | Min Value | Max Value | Unique Values | Status | Root Cause |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `density_3x3` | Spatial Density | 0.0 | 0.0 | 1 | **DEAD (0 variance)** | Filtered dataset passed to `compute_spatial_features` without unfiltered wafer map |
| `density_5x5` | Spatial Density | 0.0 | 0.0 | 1 | **DEAD (0 variance)** | Filtered dataset passed to `compute_spatial_features` without unfiltered wafer map |
| `longest_run_104` | Block Anomaly | 0.0 | 48.0 | 42 | Pruned by LightGBM | Zero split count in Model B due to colsample subsampling & collinearity with `cluster_area` |
| `cluster_width` | Block Anomaly | 0.0 | 185.0 | 156 | Pruned by LightGBM | Zero split count in Model B; `cluster_area` subsumes linear width |
| `b_min` | Block Scalar | 42.1 | 68.3 | 24,102 | Pruned by LightGBM | Zero split count; defects are positive excursions ($>104.0$), not negative dips |

**Audit Conclusion on Datasets:** The data underlying Cerebrum is genuine semiconductor synthetic data based on real WM-811K topologies with strict zero-leakage splits. However, spatial density features were accidentally zeroed out due to preprocessing call sequence errors, leaving valuable spatial clustering signal completely unexploited.
