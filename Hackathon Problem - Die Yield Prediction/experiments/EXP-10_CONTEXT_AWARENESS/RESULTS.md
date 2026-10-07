# EXP-10: Context-Aware Die Intelligence — Experimental Results

**Experiment:** EXP-10 Context-Aware Die Intelligence  
**Platform:** SanDisk Cerebrum Die Yield Prediction  
**Status:** Evaluation Completed & Audited  
**Baseline:** Frozen Production B2 Selective Cascade  
**Date:** October 2026  

---

## 1. Executive Summary & Verdict

### Final Classification: **REJECT (KEEP FROZEN B2 AS PRODUCTION CONTROL)**

In EXP-10, we investigated five distinct context families designed to evaluate dies in the context of their physical neighbourhood (Family A), spatial cluster morphology (Family B), wafer-relative baseline shifts (Family C), 2,000-block signal geometry (Family D), and multi-sensor evidence concordance (Family E).

### Core Findings:
1. **None of the context-aware candidate models surpassed the Frozen B2 baseline on the complete selective cascade.**
2. **C1 (Local Physical Neighbourhood)** came closest, achieving a marginal gain in raw validation defect recall ($37.31\% \to 37.50\%$, $+2$ rescued defects), but suffered precision dilution ($89.34\% \to 88.00\%$) and degraded performance on **14 out of 32 validation wafers**.
3. On the frozen independent holdout (`input/test.csv`, 40 unseen wafers, 32,598 dies), C1 degraded Cascade F1 from **0.5177 down to 0.5159** and recovered **3 fewer net defects** (+34 vs +37).
4. **Families B, C, D, E, C6, and C7 all suffered measurable performance degradation**, with Cascade F1 dropping by up to $-0.0170$ and Net TP recovery dropping by up to $31\%$.

---

## 2. Complete Cascade Ablation Matrix

Evaluated under locked production gating ($\tau_A = 0.5100, \tau_B = 0.5376$, Gate $[0.295, 0.550]$) on the canonical 32-wafer validation holdout (27,972 dies):

| Candidate | Num Feats | Cascade F1 | Cascade PR-AUC | Cascade Precision | Cascade Recall | ΔF1 | ΔPR-AUC | Net TP Gain | New FP | Wafer Robustness (Imp/Tie/Deg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0_Control (Frozen B2)** | **536** | **0.52639** | **0.47786** | **89.34%** | **37.31%** | **0.000** | **0.000** | **+45** | **32** | **CONTROL (0/32/0)** |
| **C1_Neighbourhood** | 543 | 0.52590 | 0.47504 | 88.00% | 37.50% | -0.00049 | -0.00282 | +47 | 38 | 8 / 10 / 14 |
| **C2_SpatialTopology** | 542 | 0.51095 | 0.47463 | 85.37% | 36.46% | -0.01544 | -0.00323 | +36 | 52 | 1 / 14 / 17 |
| **C3_WaferRelative** | 544 | 0.50938 | 0.46574 | 87.16% | 35.98% | -0.01701 | -0.01212 | +31 | 41 | 7 / 11 / 14 |
| **C4_SignalGeometry** | 544 | 0.51936 | 0.47662 | 88.01% | 36.84% | -0.00703 | -0.00124 | +40 | 39 | 8 / 13 / 11 |
| **C5_EvidenceAgreement** | 541 | 0.51364 | 0.47229 | 86.35% | 36.55% | -0.01275 | -0.00557 | +37 | 47 | 5 / 12 / 15 |
| **C6_AllContextFamilies** | 570 | 0.50960 | 0.46487 | 84.61% | 36.46% | -0.01679 | -0.01299 | +36 | 57 | 6 / 11 / 15 |
| **C7_StrongestSelected** | 544 | 0.51200 | 0.46327 | 86.49% | 36.36% | -0.01439 | -0.01459 | +35 | 46 | 7 / 13 / 12 |

---

## 3. Forensic Analysis: Why Did Context Families Underperform?

### 1. The Wafer-Relative Context Failure (Family C)
- Feature importance analysis revealed that LightGBM heavily prioritized wafer-level aggregates:
  - `wafer_mean_f171` achieved **55,152 gain** (highest among all new features).
  - `wafer_std_f171` achieved **48,001 gain** (second highest).
- **Failure Mechanism**: Because defect incidence is non-uniformly distributed across wafers (wafer defect rates vary from $0.49\%$ to $14.82\%$), the model learned to use `wafer_mean_f171` as an indirect proxy for training wafer identity. On unseen validation and test wafers, these split thresholds failed to generalize, dropping Cascade F1 by **-0.0170** and cutting rescued defects by **31%**.

### 2. Spatial Topology Redundancy (Family B)
- Existing B2 features already include `density_3x3` and `density_5x5` over the pre-test reject map (`old_label`).
- Adding directional horizontal, vertical, and diagonal concentrations introduced multi-collinear split competition without adding new information regarding latent post-screening defects. F1 dropped from 0.5264 to 0.5110, while false alarms increased from 32 to 52.

### 3. Local Neighbourhood Boundary Noise (Family A)
- While `f171_local_zscore_3x3` and `f171_local_diff_3x3` succeeded in capturing local electrical contrast, they introduced edge noise on dies along wafer perimeters (where valid neighbor count drops from 8 to 2-3). This led to higher false positives (38 vs 32) and degraded 14 out of 32 validation wafers.

### 4. Signal Geometry Dispersion (Family D)
- Analog excursions in the 2,000-block signal are already effectively summarized by B2's `cluster_area_110`, `cnt_110`, `cnt_115`, and `max_roll20`.
- Adding location-dependent metrics (`anomaly_centroid`, `anomaly_early_ratio`, `anomaly_asymmetry`) forced tree splits to segment by physical block position, which varies unpredictably across flash cell memory banks, dropping Net TP gain from +45 to +40.

---

## 4. Independent Holdout Benchmark (`input/test.csv`)

To respect validation discipline (Phase 14), only the top validation candidate (**C1_Neighbourhood**) was evaluated once against the frozen baseline on the 40 unseen wafers:

| Metric | Frozen B2 Control | C1 Candidate | Delta |
| :--- | :---: | :---: | :---: |
| **Precision** | **85.22%** | 85.43% | +0.21% |
| **Recall** | **37.17%** | 36.96% | -0.21% |
| **F1-Score** | **0.5177** | 0.5159 | **-0.0017** |
| **PR-AUC** | **0.4809** | 0.4803 | **-0.0006** |
| **ROC-AUC** | **0.8078** | 0.8073 | -0.0005 |
| **Escalation Rate** | 6.22% | 6.22% | 0.00% |
| **Compute Saved** | 93.78% | 93.78% | 0.00% |
| **FNs Rescued** | **41** / 176 | 38 / 176 | **-3 FNs** |
| **Net TP Gain** | **+37** | +34 | **-3 Net TP** |
| **New FPs** | 72 | 71 | -1 FP |

---

## 5. Engineering Verdict

**RETAIN B2 AS PRODUCTION BASELINE.**
None of the proposed contextual feature additions demonstrated a robust, reproducible improvement over the frozen B2 architecture. Frozen B2 remains the most compact, generalizable, and operationally superior model for Cerebrum.
