# MODEL MERGE COMPARISON: B2 PRODUCTION BASELINE VS TEAMMATE CANDIDATE

**Platform:** SanDisk Cerebrum Die Yield Intelligence & Cascade Optimization Platform  
**Evaluation Scope:** Head-to-Head Comparison Across Canonical Holdout and Independent EXP-04  
**Date:** October 7, 2026  
**Auditor / Integrator:** Principal Release Integration Engineer  

---

## Executive Summary

The teammate's experimental implementation from `To_be_added/` was subjected to a rigorous, controlled evaluation against the frozen production **B2 Selective Cascade Release**.

Both systems were evaluated under identical conditions across two distinct, non-overlapping validation populations:
1. **Canonical 32-Wafer Holdout** ($N = 27,972$ dies, 1,056 actual defects, 3.775% prevalence) — the dataset from which the teammate's screenshot claims originated.
2. **Independent EXP-04 Test Set (`input/test.csv`)** ($N = 32,598$ dies, 1,380 actual defects, 4.233% prevalence across 40 unseen wafers) — the authoritative production generalization gate.

---

## 1. Primary Population: Independent EXP-04 Generalization Gate

The independent EXP-04 dataset represents 40 unseen manufacturing wafers with ground-truth test outcomes. **No hyperparameters or thresholds were tuned on this dataset.**

### A. Complete Cascade Head-to-Head Comparison (Authoritative)

| Metric | Current Production B2 | Teammate Candidate (Enhanced) | Teammate Candidate (High-Recall) | Delta (Cand vs B2) | Superior Candidate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cascade Architecture** | LightGBM Selective Cascade | LightGBM + 1D-CNN Late Fusion | LightGBM + 1D-CNN Late Fusion | Architecture shift | B2 (Simpler, pure tabular) |
| **Gate Thresholds** | $[0.295, 0.550]$ | $[0.250, 0.550]$ | $[0.200, 0.500]$ | Gate widened | Candidate explored wider gates |
| **Decision Thresholds** | $\tau_A=0.5100, \tau_B=0.5376$ | $\tau_A=0.5100, \tau_B=0.5200$ | $\tau_A=0.5100, \tau_B=0.5000$ | $\tau_B$ shifted | B2 calibrated |
| **F1-Score** | **0.5177** | **0.5264** | **0.5271** | $+0.0087$ ($+1.68\%$) | **Candidate (+0.0087)** |
| **F2-Score** | **0.4190** | **0.4246** | **0.4335** | $+0.0056$ ($+1.34\%$) | **Candidate (+0.0056)** |
| **Matthews Corr (MCC)** | **0.5517** | **0.5636** | **0.5531** | $+0.0119$ ($+2.16\%$) | **Candidate (+0.0119)** |
| **Precision** | **85.22%** | **87.67%** | **82.31%** | $+2.45\%$ | **Candidate (+2.45%)** |
| **Recall** | **37.17%** | **37.61%** | **38.77%** | $+0.44\%$ | **Candidate (+0.44%)** |
| **PR-AUC** | **0.4808** | **0.5004** | **0.5092** | $+0.0196$ ($+4.08\%$) | **Candidate (+0.0196)** |
| **ROC-AUC** | **0.8078** | **0.8270** | **0.8291** | $+0.0192$ ($+2.38\%$) | **Candidate (+0.0192)** |
| **True Positives (TP)** | **513** | **519** | **535** | $+6$ TP caught | **Candidate (+6 TP)** |
| **False Positives (FP)** | **89** | **73** | **115** | $-16$ FP (False alarms cut!) | **Candidate (-16 FP)** |
| **False Negatives (FN)**| **867** | **861** | **845** | $-6$ FN | **Candidate (-6 FN)** |
| **True Negatives (TN)** | **31,129** | **31,145** | **31,103** | $+16$ TN | **Candidate (+16 TN)** |
| **Compute Saved %** | **93.78%** | **96.05%** | **92.63%** | $+2.27\%$ compute saved | **Candidate (+2.27%)** |
| **Escalation Rate %** | **6.22%** (2,028 dies) | **3.95%** (1,288 dies) | **7.37%** (2,403 dies) | $-2.27\%$ ($740$ fewer dies) | **Candidate (Fewer escalations)** |
| **Brier Score** | **0.0442** | **0.0314** | **0.0311** | $-0.0128$ (better) | **Candidate** |
| **ECE Calibration** | **0.1214** | **0.0590** | **0.0578** | $-0.0624$ (51.4% better) | **Candidate** |
| **Model A Latency** | **0.0076 ms/die** | **0.0321 ms/die** | **0.0321 ms/die** | $+0.0245$ ms/die | **B2 (4.2x faster)** |
| **Model B Latency** | **0.0068 ms/die** | **0.0557 ms/die** | **0.0557 ms/die** | $+0.0489$ ms/die | **B2 (8.2x faster)** |

---

## 2. Secondary Population: Canonical 32-Wafer Holdout Reproduction

This population verifies the numbers reported in the teammate's screenshot ($N = 27,972$, $1,056$ defects).

| Metric | Current Production B2 | Teammate Candidate (Enhanced) | Teammate Candidate (High-Recall) | Delta (Cand vs B2) | Analysis / Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F1-Score** | **0.5254** | **0.5291** | **0.5268** | $+0.0037$ | Marginally higher F1 |
| **Precision** | **89.32%** | **81.78%** | **75.22%** | **$-7.54\%$** | **Significant precision degradation** |
| **Recall** | **37.22%** | **39.11%** | **40.53%** | $+1.89\%$ | Modest recall increase |
| **PR-AUC** | **0.4777** | **0.4967** | **0.5033** | $+0.0190$ | Improved ranking |
| **ROC-AUC** | **0.7986** | **0.8255** | **0.8272** | $+0.0269$ | Higher discrimination |
| **True Positives (TP)** | **393** | **413** | **428** | $+20$ TP caught | Caught 20 additional defects |
| **False Positives (FP)** | **47** | **92** | **141** | **$+45$ FP (+95.7%)** | **Almost DOUBLE the false alarms!** |
| **False Negatives (FN)**| **663** | **643** | **628** | $-20$ FN | Reduced misses |
| **Compute Saved %** | **94.02%** | **94.68%** | **90.43%** | $+0.66\%$ | Similar compute efficiency |
| **Escalated Dies** | **1,673** | **1,489** | **2,676** | $-184$ dies | Escalation volume similar |

### Critical Takeaway on Canonical Holdout:
In the teammate's screenshot, the reported F1 was 0.5254 and Precision was 79.62% with 106 FPs. Our reproduced candidate achieves F1 = 0.5291 with 92 FPs. However, compared to B2's **47 false positives (89.32% precision)**, the teammate candidate **more than doubles false alarms (+45 to +59 FPs)** on the canonical holdout. In semiconductor manufacturing, a 10% drop in precision causes substantial unnecessary secondary testing overhead on clean dies.

---

## 3. Standalone Model A Comparison

Model A operates on parametric electrical and spatial context without expensive block waveform parsing.

| Metric | Baseline B2 Model A (506 Feats) | Candidate Model A Enhanced (518 Feats) | Delta | Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Feature Count** | 506 (500 parametric + 6 spatial) | 518 (500 parametric + 11 spatial + 7 PCA/moments) | $+12$ features | Higher complexity (PCA fit required) |
| **PR-AUC (EXP-04)** | 0.4662 | 0.4856 | $+0.0194$ | **Improved (+4.16%)** |
| **ROC-AUC (EXP-04)**| 0.8043 | 0.8250 | $+0.0207$ | **Improved (+2.57%)** |
| **F1-Score (EXP-04)**| 0.5005 | 0.5123 | $+0.0118$ | **Improved (+2.36%)** |
| **Precision (EXP-04)**| 91.19% | 91.43% | $+0.24\%$ | Preserved |
| **Recall (EXP-04)** | 34.49% | 35.58% | $+1.09\%$ | Caught 15 more defects (491 vs 476) |
| **False Positives** | 46 | 46 | **0** | **Identical false positive count** |
| **Inference Speed** | 0.0076 ms/die (131,500 dies/sec) | 0.0321 ms/die (31,100 dies/sec) | $+0.0245$ ms/die | 4.2x slower due to PCA transform |

---

## 4. Standalone Model B Comparison

Model B operates on escalated dies with full 2,000-block sequence context.

| Metric | Baseline B2 Model B (536 Feats) | Candidate Model B (529 Feats + 1D-CNN) | Delta | Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Model Type** | LightGBM Tabular | LightGBM + 1D-CNN Late Fusion ($0.90/0.10$) | Architecture shift | Requires PyTorch runtime |
| **Feature Count** | 536 tabular features | 529 tabular + 2,000-sample sequence stream | New data modality | Dual-stream late fusion |
| **PR-AUC (EXP-04)** | 0.4879 | 0.5306 | $+0.0427$ | **Substantial improvement (+8.75%)** |
| **ROC-AUC (EXP-04)**| 0.8458 | 0.8653 | $+0.0195$ | **Improved (+2.31%)** |
| **F1-Score (EXP-04)**| 0.4729 | 0.5201 | $+0.0472$ | **Substantial improvement (+9.98%)** |
| **Precision (EXP-04)**| 75.20% | 86.93% | $+11.73\%$ | **Substantial precision improvement** |
| **Recall (EXP-04)** | 34.49% | 37.10% | $+2.61\%$ | Caught 36 more defects (512 vs 476) |
| **False Positives** | 157 | 77 | **$-80$ FP** | **Cut standalone false positives by 51%!** |
| **Inference Speed** | 0.0068 ms/die | 0.0557 ms/die | $+0.0489$ ms/die | 8.2x slower due to PyTorch CNN forward pass |

---

## 5. Feature Provenance & Leakage Audit

A comprehensive leakage audit was conducted across all 23 new features introduced by the teammate:

1. **Polar Spatial Features (11 features):**
   - `radial_dist`, `sin_theta`, `cos_theta`, `edge_dist`, `die_row`, `die_col`: Calculated purely from die coordinates relative to wafer center and wafer boundaries. **Zero leakage.**
   - `density_3x3`, `density_5x5`, `density_7x7`, `wafer_old_fail_rate`: Computed exclusively from `old_label` (pre-test / probe fail outcomes from sorting test CP1). Because sorting fails are physical ground truth prior to downstream functional testing, this information is available at inference time. **Zero post-test leakage.**
2. **Multivariate Statistical & PCA Moments (7 features):**
   - Standardized against the mean and variance of **passing dies in the training split only**.
   - PCA ($k=25$) fitted strictly on **training passing dies**.
   - `param_max_abs_z`, `param_mean_abs_z`, `param_std_z`, `param_sum_sq_z`, `param_cnt_gt_2sigma`, `param_cnt_gt_3sigma`, `pca_recon_error`: Computed per-die from parametric features $1..500$. **Zero future leakage.**
3. **Block Sequence Physics Features (11 features):**
   - `b_mean`, `b_std`, `b_min`, `b_max`, `b_p95`, `b_p99`, `b_cnt_104`, `b_cnt_106`, `b_max_spike_dev`, `b_first_diff`, `b_longest_run_104`: Computed exclusively from the 2,000 blocks of the specific die being evaluated. **Zero inter-die leakage.**

**Audit Conclusion:** The teammate's candidate features are 100% physically valid, mathematically sound, and free from any temporal, cross-wafer, or label leakage.

---

## 6. Wafer-Level Robustness

On the 40 unseen wafers of EXP-04:

| Metric | Production B2 Cascade | Candidate Enhanced Cascade | Assessment |
| :--- | :--- | :--- | :--- |
| **Wafer F1 Mean** | 0.4879 | 0.4952 | Candidate is slightly more robust |
| **Wafer F1 Std Dev** | 0.1707 | 0.1648 | Lower variance across wafers |
| **Wafer F1 Min (Worst wafer)** | 0.0000 | 0.0000 | Both struggle on 1 extreme low-defect wafer |
| **Wafer F1 Max (Best wafer)** | 1.0000 | 1.0000 | Identical peak performance |
| **Wafers with High False Alarms (FP > 5)** | 4 wafers | 3 wafers | Candidate reduces high-noise wafers |

---

## 7. Latency and Operational Tradeoffs

| Operational Dimension | Current Production B2 | Teammate Candidate (Enhanced) | Operational Assessment |
| :--- | :--- | :--- | :--- |
| **Fast-Path Latency (Model A)** | 0.0076 ms | 0.0321 ms | Both easily satisfy <1.0 ms tester budget |
| **Escalated Latency (Model B)** | 0.0068 ms | 0.0557 ms | PyTorch CNN batch forward pass adds ~0.05 ms |
| **Effective Cascade Latency** | 0.0080 ms/die | 0.0343 ms/die | ~29,100 dies/second throughput |
| **Memory Footprint** | ~5 MB LightGBM models | ~5 MB LightGBM + 175 KB PyTorch CNN + 57 KB PCA | Very compact (<10 MB total) |
| **Runtime Dependencies** | Python, NumPy, LightGBM | Python, NumPy, LightGBM, PyTorch, SciPy | PyTorch required in production inference |
| **Escalation Volume on 32.5k dies**| 2,028 dies (6.22%) | 1,288 dies (3.95%) | Candidate saves $740$ secondary test reads |

---

## Summary Matrix: B2 vs Candidate

```
                            EXP-04 PR-AUC      EXP-04 F1       EXP-04 Precision   EXP-04 FP   Compute Saved   Runtime Dep
Production B2 Cascade:          0.4808           0.5177             85.22%            89         93.78%       LightGBM only
Candidate Enhanced Cascade:     0.5004           0.5264             87.67%            73         96.05%       LightGBM + PyTorch
Delta:                         +0.0196          +0.0087             +2.45%           -16         +2.27%       
```
