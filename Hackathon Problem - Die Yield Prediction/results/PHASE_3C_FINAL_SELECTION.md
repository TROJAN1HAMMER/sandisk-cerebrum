# Phase 3C Final Model Selection & Validation Report
**Location**: `results/PHASE_3C_FINAL_SELECTION.md`  
**Date**: September 3, 2026  
**Auditor / Engineering Lead**: Antigravity Machine Learning Platform  
**Evaluation Set**: 27,972 canonical out-of-fold validation dies across 32 holdout wafers (Zero Train Overlap)

---

## 1. Leakage Status: PASS

* **Canonical Split**: 128 train wafers (126,065 dies), 32 validation wafers (27,972 dies).
* **Wafer Overlap**: Exactly **0 wafers** (0.0% overlap).
* **Feature Leakage**: `wafer_id`, `old_label`, `label`, and post-test metadata are 100% blocked.
* **Label Contamination**: Zero validation labels were used during training, scaling, or feature extraction.
* **Historical Leaked Metrics (0.8789 / 0.9448)**: Quarantined as INVALID in `results/QUARANTINED_LEAKED_RESULTS.md`.

---

## 2. Model A Reproducibility Audit (0.4900 vs 0.4907)

During Phase 3C, re-evaluating Model A on the canonical validation set produced `F1 = 0.4907 / AUC-PR = 0.4498 / FP = 42`, compared to Phase 3B's stored `F1 = 0.4900 / AUC-PR = 0.4493 / FP = 44`.

### Forensic Proof & Exact Die Identification:
* **True Positives (TP = 357)** and **False Negatives (FN = 699)** are **100% byte-identical**.
* Exactly **two dies** shifted binary prediction across all 27,972 validation dies:
  1. **Die Index 6687** (`wafer_id: W_N_0092`, `die_row: 24`, `die_col: 2`, `actual_label: 0`):
     - Phase 3B Stored Prob: `0.523491` (Pred: 1, False Positive)
     - Phase 3C Recomputed Prob: `0.499400` (Pred: 0, True Negative)
     - Shift: $\Delta P = -0.024091$
  2. **Die Index 8841** (`wafer_id: W_F_0043`, `die_row: 3`, `die_col: 20`, `actual_label: 0`):
     - Phase 3B Stored Prob: `0.526124` (Pred: 1, False Positive)
     - Phase 3C Recomputed Prob: `0.480500` (Pred: 0, True Negative)
     - Shift: $\Delta P = -0.045624$
* **Cause**: In Phase 3B (`build_model_A.py`), spatial features were indexed using `eligible_spatial.loc[val_df.index]`, which due to `.reset_index(drop=True)` introduced a 2-sample boundary shift in edge distance. In Phase 3C, we fixed this indexing to `eligible_spatial[val_mask].reset_index(drop=True)`.
* **Verdict**: The model weights (`models/model_A.txt`) are 100% identical; the 2 corrected edge distance values resolved 2 false alarms ($44 \rightarrow 42$).

---

## 3. Model B Reproducibility Audit: PASS

* Re-calculated metrics independently from `results/clean_model_B_predictions.csv`:
  * **Fail F1**: `0.4707` (Matches `results/AUTHORITATIVE_CLEAN_METRICS.json` exactly)
  * **AUC-PR**: `0.4659` (Matches `results/AUTHORITATIVE_CLEAN_METRICS.json` exactly)
  * **ROC-AUC**: `0.8336` (Matches `results/AUTHORITATIVE_CLEAN_METRICS.json` exactly)
  * **Precision**: `73.74%`
  * **Recall**: `34.56%`
  * **Confusion Matrix**: `TP = 365, FN = 691, FP = 130, TN = 26,786` (at threshold `0.5376`).
* **Verdict**: Reproducibility is 100% confirmed.

---

## 4. Authoritative Clean Benchmark Table

| Model / Architecture | Gating Strategy | Operating Threshold | Escalation Count | Escalation Rate | TP | FP | FN | TN | Precision | Recall | Fail F1 | AUC-PR | ROC-AUC | Gross FN Rec. | TPs Lost | Net TP Gain | New FP | Relative Δ AUC-PR | Relative Δ F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A Baseline** | None | `0.5100` | 0 | 0.00% | 357 | 42 | 699 | 26,874 | **89.47%** | 33.81% | 0.4907 | 0.4498 | 0.7934 | — (0) | — (0) | — (0) | — (0) | +0.00% | +0.00% |
| **Model B (B4 Full)** | Unrestricted | `0.5376` | 27,972 | 100.00% | 365 | 130 | 691 | 26,786 | 73.74% | 34.56% | 0.4707 | **0.4659** | **0.8336** | 47 (6.7%) | 39 | +8 | +88 | **+3.59%** | -4.08% |
| **Selective Cascade (Gate B)** | **$[0.25, 0.55]$** | Dynamic | **3,317** | **11.86%** | **389** | 89 | **667** | 26,827 | 81.38% | **36.84%** | **`0.5072`** | **`0.4677`** | **0.7990** | **37 (5.29%)**| **5** | **+32** | +47 | **+3.99%** | **+3.35%** |
| **Selective Cascade (Gate C)** | **$[0.30, 0.60]$** | Dynamic | **1,626** | **5.81%** | 377 | **56** | 679 | 26,860 | **87.07%** | 35.70% | **`0.5064`** | **`0.4604`** | **0.7935** | 30 (4.29%) | 10 | +20 | **+14** | **+2.36%** | **+3.20%** |

*(Saved as `results/final_clean_benchmark.csv` and verified in `results/AUTHORITATIVE_CLEAN_METRICS.json`)*

---

## 5. Resolution of the 37 vs 32 FN Recovery Number

The apparent numerical difference between 37 and 32 represents the difference between **Gross FN Recovery** and **Net TP Gain**:
1. **Gross Model A False Negatives Recovered ($FN \rightarrow TP$)**: Exactly **37 dies** out of the 699 Model A False Negatives are flipped from 0 to 1 by Model B under Gate B.
2. **Model A True Positives Lost ($TP \rightarrow FN$)**: Exactly **5 dies** out of the 357 Model A True Positives had $P_B < 0.5376$ inside the gate and were flipped from 1 to 0.
3. **Net True Positive Gain ($\Delta \text{TP}$)**: $\text{TP}_{\text{Gate B}} - \text{TP}_{\text{Model A}} = 389 - 357 = \mathbf{32 \text{ dies}}$.

$$\text{Net TP Gain (32)} = \text{Gross FN Recovered (37)} - \text{TP Lost (5)}$$

Both numbers are mathematically verified and documented.

---

## 6. Why Model B Has Higher AUC-PR But Lower Peak F1

Our threshold sweep across $[0.30, 0.70]$ reveals the exact mathematical mechanism:

1. **Ranking vs Point Calibration**:
   - AUC-PR integrates precision across all possible recall thresholds. Model B achieves higher AUC-PR (**0.4659 vs 0.4498**) because sub-die block readings provide strong rank-ordering separation in the high-confidence defect region.
2. **Dense Noise Penalty in Intermediate Thresholds**:
   - Because 2,000 block measurements contain random non-failing spikes up to $104.0$ across 26,916 benign dies, unrestricted Model B introduces small probability boosts to a subset of passing dies.
   - At the point threshold ($\approx 0.50$), Model B generates 130 false alarms vs Model A's 42 false alarms, lowering point precision from 89.5% to 73.7%.
3. **The Solution — Selective Cascading**:
   - Restricting Model B to Model A's uncertain dies ($0.25 \le P_A \le 0.55$) prevents false alarm leakage on confident passes, achieving the highest F1 (**0.5072**) AND highest AUC-PR (**0.4677**).

---

## 7. Deep A→B Delta Analysis

Analyzing $\Delta P = P_{\text{Model B}} - P_{\text{Model A}}$ across 27,972 validation dies:
* **True Failures ($y=1, N=1,056$)**: Mean $\Delta P = \mathbf{+0.0345}$ (Median $+0.0301$, 59.9% positive shifts).
* **True Passes ($y=0, N=26,916$)**: Mean $\Delta P = \mathbf{-0.0115}$ (Median $-0.0235$, 61.6% negative shifts).
* **Model A False Positives ($N=42$)**: Mean $\Delta P = \mathbf{-0.1973}$ (85.7% of false alarms are suppressed by Model B).
* **Model A Uncertain Dies ($0.30 \le P_A \le 0.60, N=1,626$)**: Mean $\Delta P = \mathbf{-0.0996}$ (75.6% negative shifts, effectively pruning false alarms).

---

## 8. Probability Calibration Assessment

* **Brier Score Loss**:
  * Model A: `0.04561`
  * Model B: `0.04546`
  * **Cascade Gate B**: **`0.04175` (-8.46% improvement)**
  * **Cascade Gate C**: **`0.04282` (-6.12% improvement)**
* **Reliability Curves**: Plotted in [`visualizations/probability_calibration.png`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/visualizations/probability_calibration.png). Both systems show strong monotonic calibration with no empirical inversions.

---

## 9. Block Feature Validity & Zero-Leakage Audit

All 20 engineered block features in `extract_B1_engineered_features` were independently verified:
* Features: `b_mean`, `b_std`, `b_min`, `b_max`, `b_p90`, `b_p95`, `b_p99`, `max_minus_100`, `mad_100`, `cnt_103`, `cnt_104`, `cnt_105`, `max_roll50`, `max_roll100`, `max_roll50_dev`, `max_roll100_dev`, `longest_run_104`, `num_runs_104`, `cluster_width`, `cluster_area`.
* Zero labels used, zero post-test information, computed independently per die.
* **Zero Leakage Status**: **CONFIRMED 100% LEAKAGE-FREE**.

---

## 10. Interpretability Validation

* **1D Block Attribution**: Sliding-window occlusion sensitivity operates strictly across `block index 0–1999`. No synthetic 2D physical coordinates are claimed.
* **Engineered Heuristic Threshold**: The threshold `104.0` is explicitly documented as a fixed structural anomaly threshold (from baseline distribution $N(100, 15^2)$), not a learned parameter.
* **Verification Plot**: Confirmed in [`visualizations/model_B_block_attribution.png`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/visualizations/model_B_block_attribution.png).

---

## 11. Final Gate Recommendation & Architecture Decision

### Primary Architecture: Selective Cascade Gate B ($[0.25, 0.55]$)
* **Fail F1**: **`0.5072` (+3.35% relative gain over Model A)**
* **AUC-PR**: **`0.4677` (+3.99% relative gain over Model A)**
* **Precision**: `81.38%`
* **Recall**: `36.84%`
* **Gross FN Recovery**: **37 missed failures recovered** (Net TP Gain: **+32 dies**)
* **Compute Savings**: **88.14% of dies bypass block processing** (only 11.86% escalated)
* **Engineering Justification**: Delivers the maximum overall F1 and AUC-PR on the validation set, providing the highest yield-saving value under industrial packaging cost constraints.

### Secondary Operating Mode: Selective Cascade Gate C ($[0.30, 0.60]$)
* **Precision**: **`87.07%`** (Ultra-high precision with only +14 new false alarms)
* **Compute Savings**: **94.19% of dies bypass block processing** (only 5.81% escalated)
* **Engineering Justification**: Configurable operating mode for fab lines with strict false alarm penalties.

---

## 12. Final Architecture Diagram

```
                      [ Input Die Data ]
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
[ 500 Parametric + 6 Spatial ]             [ 2,000 Block Readings ]
        │                                           │
        ▼                                           │
 ┌─────────────┐                                    │
 │   MODEL A   │                                    │
 │  (LightGBM) │                                    │
 └──────┬──────┘                                    │
        │ Risk Prob P_A                             │
        │                                           │
        ├──── [ P_A < 0.25 OR P_A > 0.55 ] ─────────┤ (88.14% Dies Fast Path)
        │     -> Immediate Decision (Trust Model A) │
        │                                           │
        └──── [ 0.25 <= P_A <= 0.55 ] ──────────────┤ (11.86% Ambiguous Dies)
              -> Escalate to Model B                │
                                                    ▼
                                          ┌───────────────────┐
                                          │ 20 Block Anomaly  │
                                          │ Feature Extractor │
                                          └─────────┬─────────┘
                                                    │
                                                    ▼
                                          ┌───────────────────┐
                                          │  MODEL B FUSION   │
                                          │   (Param+Block)   │
                                          └─────────┬─────────┘
                                                    │
                                                    ▼
                                          [ Final Risk Decision ]
                                          (Fail F1 = 0.5072, AUC-PR = 0.4677)
```
