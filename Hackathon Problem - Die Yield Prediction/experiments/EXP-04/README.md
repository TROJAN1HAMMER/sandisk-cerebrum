# EXP-04: Independent Holdout Generalization & Final Model Validation

## Overview
**EXP-04** is the final independent holdout validation experiment for the SanDisk Cerebrum Die Yield Prediction platform.

Following the confirmation and promotion of candidate **B2 (Enhanced Model B)** in **EXP-03B**, this experiment rigorously evaluates the locked, frozen cascade system on an unseen, independent population that has never participated in model training, feature selection, threshold search, or candidate forensics: the 40 unseen wafers of `input/test.csv`.

## Core Validation Principles
1. **Zero Tuning / Model Fitting**:
   - Model A: Spatial-restored LightGBM locked from EXP-01 (`experiments/EXP-01/model_A_spatial.pkl`).
   - Model B: Frozen confirmed B2 candidate from EXP-03B (`experiments/EXP-03B/models/model_B_B2_confirmed.pkl`).
   - Zero parameter fitting, zero re-training, zero threshold changes.
2. **Locked Cascade Policy (Policy B Preferred)**:
   - Model A operating threshold: $\tau_A = 0.5100$.
   - Model B operating threshold: $\tau_B = 0.5376$.
   - Escalation Gating Window: $\text{Gate B} = [0.295, 0.550]$.
3. **Independent Evaluation Population**:
   - Dataset: `input/test.csv` (39,351 raw rows, 40 unique unseen wafers).
   - Pre-test eligible dies evaluated: 32,598 dies (`old_label == 0`).
   - Actual ground-truth latent failures: 1,380 dies (4.2334% defect rate).
   - Zero wafer overlap: $\text{Train Wafers (128)} \cap \text{Val Wafers (32)} \cap \text{Test Wafers (40)} = \emptyset$.
4. **Zero Test-Label Contamination**:
   - Provenance audit verified that test labels were never accessed by any prior training or optimization algorithm.
5. **Statistical Uncertainty**:
   - 1,000 paired wafer bootstrap iterations performed to compute rigorous 95% confidence intervals.

## Summary Results

### 1. Generalization Performance (Frozen B2 Cascade)
| Evaluation Metric | Canonical Holdout (EXP-03B) | Independent Holdout (EXP-04) | Delta (EXP-04 - EXP-03B) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC** | 0.4777 | **0.4808** | **+0.0031** | **+0.65%** |
| **ROC-AUC** | 0.7986 | **0.8078** | **+0.0093** | **+1.16%** |
| **Recall** | 37.22% | **37.17%** | -0.05% | -0.11% |
| **Precision** | 89.32% | **85.22%** | -4.10% | -4.59% |
| **F1 Score** | 0.5254 | **0.5177** | **-0.0077** | **-1.47%** |
| **F2 Score** | 0.4213 | **0.4190** | **-0.0023** | **-0.55%** |
| **MCC** | 0.5674 | **0.5517** | -0.0157 | -2.77% |
| **Brier Score** | 0.0417 | **0.0442** | +0.0025 | +5.98% |
| **ECE** | 0.1232 | **0.1214** | **-0.0018** | **-1.43%** |
| **True Positives (TP)** | 393 | **513** | +120 | +30.53% |
| **False Negatives (FN)**| 663 | **867** | +204 | +30.77% |
| **False Positives (FP)**| 47 | **89** | +42 | — |
| **True Negatives (TN)** | 26,869 | **31,129** | +4,260 | +15.85% |
| **Escalation Rate** | 5.98% | **6.22%** | +0.24% | +4.09% |
| **Compute Saved** | 94.02% | **93.78%** | -0.24% | -0.26% |

### 2. Incremental Value Over Model A Alone on Independent Data
| Metric | Model A Alone ($\tau_A = 0.5100$) | Frozen B2 Cascade | Cascade Improvement |
| :--- | :---: | :---: | :---: |
| **F1 Score** | 0.5005 | **0.5177** | **+0.0172 (+3.44% rel)** |
| **F2 Score** | 0.3939 | **0.4190** | **+0.0251 (+6.37% rel)** |
| **Recall** | 34.49% | **37.17%** | **+2.68% (+7.77% rel)** |
| **Precision** | 91.19% | **85.22%** | -5.97% |
| **PR-AUC** | 0.4662 | **0.4808** | **+0.0146 (+3.13% rel)** |
| **ROC-AUC** | 0.8043 | **0.8078** | **+0.0035 (+0.44% rel)** |
| **True Positives (TP)** | 476 | **513** | **+37 Net Defect Gain** |
| **False Negatives (FN)**| 904 | **867** | **-37 Defect Misses** |
| **Escalation Rate** | 0.00% | **6.22%** | +2,028 dies evaluated |
| **Compute Saved** | 100.00% | **93.78%** | 30,570 dies fast-path |

### 3. Paired Wafer Bootstrap 95% Confidence Intervals (1,000 resamples)
- **Cascade F1**: Point Estimate = **0.5177** | 95% CI = **[0.4948, 0.5369]**
- **Cascade F2**: Point Estimate = **0.4190** | 95% CI = **[0.3995, 0.4403]**
- **Cascade MCC**: Point Estimate = **0.5517** | 95% CI = **[0.5260, 0.5701]**
- **Cascade PR-AUC**: Point Estimate = **0.4808** | 95% CI = **[0.4465, 0.5089]**

## Final Status
```text
FINAL VALIDATED CANDIDATE
```
The frozen B2 selective cascade successfully demonstrates robust generalization beyond the canonical holdout:
1. The generalization gap on the independent population is minimal (**-0.0077 F1, -1.47% relative**), with point estimate $0.5177$ cleanly inside the bootstrap CI $[0.4948, 0.5369]$.
2. The cascade delivers a verified **+37 net defect recovery** (+7.77% relative recall gain) over Model A alone while preserving **93.78% compute savings**.
3. Zero data leakage, 100% wafer isolation, and zero production artifact mutation verified.
