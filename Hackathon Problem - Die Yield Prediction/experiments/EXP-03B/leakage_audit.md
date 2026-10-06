# EXP-03B Feature Leakage & Provenance Audit

## Objective
Establish complete mathematical and temporal proof that every feature used in candidate models **B2** and **B3** is strictly leakage-free, causal, and computable prior to post-test inspection.

---

## 1. Provenance & Isolation Matrix

| Feature | Input Source | Available at Inference? | Uses Target `label`? | Uses Neighbor `label`? | Uses Post-Test Info? | Leakage Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `feature_1` ... `feature_500` | In-line parametric wafer-fab metrology | YES | NO | NO | NO | **CLEAN** |
| `radial_dist`, `edge_dist`, `die_row`, `die_col` | Wafer physical grid coordinates | YES | NO | NO | NO | **CLEAN** |
| `density_3x3_restored` | Pre-test `old_label` on wafer grid | YES | NO | NO | NO | **CLEAN** |
| `density_5x5_restored` | Pre-test `old_label` on wafer grid | YES | NO | NO | NO | **CLEAN** |
| Baseline block features (`b_mean`, `b_std`, etc.) | Pre-test 2,000 raw sensor readings | YES | NO | NO | NO | **CLEAN** |
| `cnt_110`, `cnt_115`, `cnt_120` | Raw sensor readings of current die | YES | NO | NO | NO | **CLEAN** |
| `cluster_area_110`, `cluster_area_115` | Raw sensor readings of current die | YES | NO | NO | NO | **CLEAN** |
| `max_roll20`, `max_roll20_dev`, `max_roll20_rel` | Raw sensor readings of current die | YES | NO | NO | NO | **CLEAN** |
| `peak_prominence` | Raw sensor readings of current die | YES | NO | NO | NO | **CLEAN** |
| `top10_dev` | Raw sensor readings of current die | YES | NO | NO | NO | **CLEAN** |

---

## 2. Rigorous Verification of Spatial Defect Densities

* **Source Column:** Strictly `old_label` (which indicates whether a die failed during preliminary pre-test wafer inspection).
* **Target Isolation:** The target column `label` (final post-test electrical defect outcome) was **never accessed** during neighborhood convolution.
* **Wafer Isolation:** Spatial convolutions are executed strictly wafer-by-wafer on 2D coordinates `(die_row, die_col)`. No information crosses wafer boundaries.
* **Zero Temporal Leakage:** `old_label` represents observations made during front-end wafer inspection, temporally preceding both Model A and Model B inference.

---

## 3. Rigorous Verification of Block Sensor Features

* **Independence:** All 10 candidate block features are computed entirely **within the single die's 1D signal vector** of 2,000 readings.
* **Zero Cross-Die Coupling:** No summary statistics, percentiles, or normalizations borrow data from other dies, other wafers, or the validation split.
* **Deterministic Transformation:** Every feature transformation is a deterministic mathematical operator (counting, integration, 1D convolution, percentiles).

---

## Audit Verdict
**100% LEAKAGE-FREE**: Every feature in B2 and B3 satisfies full semiconductor fab production causal validity.
