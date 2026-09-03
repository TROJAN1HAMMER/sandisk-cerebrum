# Split & Evaluation Protocol Hardening Report
**Location**: `results/SPLIT_HARDENING_REPORT.md`  
**Date**: September 3, 2026  
**Status**: `HARDENING COMPLETE & VERIFIED`

---

## 1. Root Cause Fixed

During the forensic audit, an RNG protocol discrepancy was uncovered:
* In Phase 3B (`build_model_A.py`), Model A was trained on 128 wafers split with:
  `np.random.seed(42); shuffled_wafers = np.random.permutation(unique_wafers)`
* In Phase 3C (`src/data_loader.py`), the split helper was rewritten using:
  `rng = np.random.default_rng(42); shuffled_wafers = rng.permutation(unique_wafers)`

Because `np.random.default_rng` uses a completely different permutation sequence, 28 of the 32 "validation" wafers in `src/data_loader.py` were identical to Model A's training wafers. Evaluating Model A on this overlapping set produced an artificially inflated and invalid `F1 = 0.8789 / AUC-PR = 0.9448`.

---

## 2. Canonical Split Artifact

We established `results/canonical_wafer_split.json` as the single authoritative source of truth:
* **Random Seed**: `42`
* **Total Wafers**: `160` (173,099 raw rows, 154,037 eligible dies)
* **Train Wafers**: `128` wafers (126,065 eligible dies, 5,463 failures, 4.33% fail rate)
* **Validation Wafers**: `32` wafers (27,972 eligible dies, 1,056 failures, 3.78% fail rate)
* **Train/Validation Overlap**: **Exactly 0 Wafers (0.0% intersection)**
* **Determinism**: Fully deterministic, verified via unit tests.

---

## 3. Files Changed

1. [`results/canonical_wafer_split.json`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/results/canonical_wafer_split.json): Canonical split artifact containing exact 128/32 wafer IDs and metadata.
2. [`src/data_loader.py`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/src/data_loader.py): Implemented `load_canonical_wafer_split()` and `get_wafer_split()` that read the canonical JSON and fail loudly on missing split or contamination.
3. [`src/evaluation_protocol.py`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/src/evaluation_protocol.py): New centralized validation protocol module containing `validate_eligible_dataset()`, `validate_features_not_leaked()`, `validate_wafer_partition_integrity()`, and standardized metric calculation.
4. [`build_model_A.py`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/build_model_A.py): Hardened to consume canonical split and validate partition integrity.
5. [`build_model_B.py`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/build_model_B.py): Hardened to consume canonical split and corrected spatial dataframe boolean slicing.
6. [`build_selective_cascade.py`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/build_selective_cascade.py): Hardened to consume canonical split and corrected spatial dataframe boolean slicing.
7. [`README.md`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/README.md): Updated baseline section to quote authoritative clean Model A metrics and quarantined 0.8789.
8. [`results/QUARANTINED_LEAKED_RESULTS.md`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/results/QUARANTINED_LEAKED_RESULTS.md): Formal register documenting invalid artifacts and quarantine reasoning.
9. [`tests/test_split_integrity.py`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/tests/test_split_integrity.py): Automated test suite verifying split loading, zero overlap, determinism, clean baseline match, and feature leakage assertions.

---

## 4. Files NOT Changed (Protected Baseline Artifacts)

* **`models/model_A.txt`** (1,086,850 bytes) — **UNCHANGED** (Protected locked Model A Booster).
* **`models/model_A.pkl`** (1,088,749 bytes) — **UNCHANGED** (Protected locked Model A Classifier).
* **`models/feature_metadata.json`** (43,149 bytes) — **UNCHANGED** (Protected Model A feature schema & threshold 0.51).
* **`results/val_predictions_model_A.csv`** (1,127,244 bytes) — **UNCHANGED** (Protected clean holdout predictions).
* **`generate_data.py`** & **`config.yaml`** — **UNCHANGED** (Organizer-provided assets).

---

## 5. Hard Safety Checks Summary

| Safety Check | Rule & Implementation | Result |
| :--- | :--- | :--- |
| **1. Train Wafer Uniqueness** | `len(train_wafers) == 128` and set contains distinct IDs | **PASS** |
| **2. Validation Wafer Uniqueness** | `len(val_wafers) == 32` and set contains distinct IDs | **PASS** |
| **3. Zero Train/Val Overlap** | `len(train_wafers.intersection(val_wafers)) == 0` | **PASS** |
| **4. Accounted Wafers** | All 160 dataset wafers accounted for without omission | **PASS** |
| **5. No Val Wafer in Train** | `actual_train_wafers.intersection(val_wafers) == 0` | **PASS** |
| **6. Canonical Val Membership** | `actual_val_wafers - val_wafers == set()` | **PASS** |
| **7. Canonical Train Membership**| `actual_train_wafers - train_wafers == set()` | **PASS** |
| **8. Eligible Filtering** | `(eligible_df['old_label'] == 0).all()` enforced | **PASS** |
| **9. No `wafer_id` Feature** | `wafer_id` strictly blocked from feature sets | **PASS** |
| **10. No `label` Feature** | `label` strictly blocked from feature sets | **PASS** |

---

## 6. Clean Model A Authoritative Baseline Verification

Independently verified from `results/val_predictions_model_A.csv`:
* **Holdout Validation Dies**: 27,972 dies across 32 holdout wafers
* **Positives / Negatives**: 1,056 positives (3.78%), 26,916 negatives
* **Fail F1 Score**: **`0.4900`**
* **AUC-PR**: **`0.4493`**
* **ROC-AUC**: **`0.7927`**
* **Precision**: **`89.03%`**
* **Recall**: **`33.81%`**
* **Confusion Matrix**: `TP = 357`, `FN = 699`, `FP = 44`, `TN = 26,872` (at locked threshold `0.5100`)

---

## 7. Leaked Results Status

* `results/validation_predictions.csv` (F1 = 0.8789, AUC-PR = 0.9448) is **QUARANTINED AS INVALID**.
* `results/final_comparison.csv` and `results/final_submission_metrics.csv` are **QUARANTINED AS INVALID**.

---

## 8. Remaining Work

Model B and Selective Cascade must be rebuilt and reevaluated on the canonical clean validation split in the next phase.
