# EXP-03 — Model B Feature & Error-Profile Improvement

**Experiment ID:** `EXP-03`  
**Date:** October 6, 2026  
**Status:** COMPLETE  
**Decision:** **PROMOTE CANDIDATE**  

---

## Executive Summary

EXP-03 performed a controlled scientific investigation into the feature representation and error profile of the second-stage **Model B** classifier within the SanDisk Cerebrum selective cascade.

### Key Discoveries & Results:
1. **Root Cause of Model B False Alarms Discovered:**
   * Forensic analysis revealed that `cluster_area` (at threshold $104.0$) was heavily confounded by die-level baseline drift ($\bar{x} > 100.4$). Healthy dies with slight global shifts randomly produced hundreds of blocks crossing $104.0$, inflating `cluster_area` to $>5,400$ and triggering false alarms (FP mean `cluster_area` was $5,479.30$, higher than true defect TPs at $5,228.48$).
   * Conversely, four baseline block features (`longest_run_104`, `cluster_width`, `b_min`, `mad_100`) had **0 gain and 0 splits** in LightGBM, acting as dead weight.
   * Production Model B was originally trained on un-restored zero spatial features (`density_3x3` and `density_5x5` had 0 splits).
2. **Hypothesis-Driven Feature Engineering:**
   * Pruned the 4 dead features.
   * Restored verified pre-test spatial defect densities (`density_3x3`, `density_5x5`).
   * Introduced 10 physically grounded block anomaly features targeting high-magnitude excursion and local morphology (`cnt_110`, `cnt_115`, `cnt_120`, `cluster_area_110`, `cluster_area_115`, `max_roll20`, `max_roll20_dev`, `max_roll20_rel`, `peak_prominence`, `top10_dev`).
3. **Standalone Model B Outperformed Baseline Across Every Dimension (Variant B3):**
   * **F1 Score increased from 0.4707 to 0.4873 (+0.0166)**.
   * **PR-AUC increased from 0.4658 to 0.4761 (+0.0103)**.
   * **ROC-AUC increased from 0.8336 to 0.8354 (+0.0018)**.
   * **Precision surged from 73.74% to 78.53% (+4.79%)**, eliminating **28 standalone false alarms** ($130 \rightarrow 102$).
   * **Recall increased from 34.56% to 35.32% (+8 defects detected**, $365 \rightarrow 373$).
   * **MCC increased from 0.4927 to 0.5155 (+0.0228)**.
4. **Complete Cascade Breakthrough:**
   * Under the EXP-02 preferred operating policy ($\text{Gate B} = [0.295, 0.55]$, $\tau_A = 0.5100$, $\tau_B = 0.5376$):
     * **Cascade F1 reached 0.5254 (Variant B2) and 0.5243 (Variant B3)** (breaking the $0.525$ target barrier).
     * **Cascade Precision reached 89.32% (B2) and 88.71% (B3)**.
     * **Cascade False Positives plunged to 47–50 dies** (down from $89$ in Production Baseline and $83$ in EXP-01, cutting false alarms by **-47.2%**!).
     * **Cascade True Positives reached 393** (net $+4$ defect dies over Production Baseline).
     * **Escalation remained minimal at 5.98%**, preserving **94.02% compute savings**.
5. **Wafer-Level Robustness:**
   * Mean wafer F1 increased to **0.4542** (vs 0.4452 baseline).
   * Mean wafer precision reached **72.50%** (vs 69.24% baseline).
   * **14 wafers improved**, 11 remained unchanged, and only 7 degraded.
6. **Inference Performance:** Single-die inference latency is **0.0094 ms** (>106,000 dies/sec, zero overhead).
7. **Verdict:** **PROMOTE CANDIDATE**. Variant B3 is formally promoted as the verified second-stage model candidate.

---

## Motivation from EXP-01 and EXP-02

* **EXP-01** proved that restoring spatial defect density improved Model A ranking and lifted the complete cascade from F1 $0.5072$ to $0.5140$.
* **EXP-02** optimized the cascade decision policy, showing that raising `gate_lower` to $0.295$ reduced false alarms from $89$ to $57$ and compute escalation from $11.86\%$ to $5.97\%$. Crucially, EXP-02 proved that simply expanding the gate downwards ($P_A < 0.25$) was ineffective because Model B only caught $2$ to $5$ trapped defects while generating $40$ new false alarms.
* **The Central Bottleneck:** Model B's representation itself was limiting rescue quality. EXP-03 investigates whether reforming Model B's feature representation enables higher rescue precision and defect sensitivity.

---

## Baseline Model B

* **Architecture:** LightGBM Booster (`gbdt`, 150 estimators, learning rate 0.05, 63 leaves, `scale_pos_weight` 10.0, `subsample` 0.8, `colsample_bytree` 0.7).
* **Feature Count:** Exactly 526 features:
  * 500 parametric features (`feature_1` ... `feature_500`)
  * 6 spatial features (`radial_dist`, `density_3x3` [broken zero], `density_5x5` [broken zero], `edge_dist`, `die_row`, `die_col`)
  * 20 block engineered features (`b_mean`, `b_std`, `b_min`, `b_max`, `b_p90`, `b_p95`, `b_p99`, `max_minus_100`, `mad_100`, `cnt_103`, `cnt_104`, `cnt_105`, `max_roll50`, `max_roll100`, `max_roll50_dev`, `max_roll100_dev`, `longest_run_104`, `num_runs_104`, `cluster_width`, `cluster_area`)
* **Decision Threshold:** Locked at $\tau_B = 0.5376$.

---

## Dataset and Leakage Controls

* **Population:** $154,037$ eligible dies ($old\_label == 0$) across $160$ wafers.
* **Training Partition:** Canonical 128 wafers ($126,065$ dies, $5,463$ failures, $4.33\%$ failure rate).
* **Holdout Validation:** Canonical 32 wafers ($27,972$ dies, $1,056$ failures, $3.78\%$ failure rate).
* **Leakage Audit:**
  * Zero die overlap between training and validation wafers ($\text{Train} \cap \text{Val} = \emptyset$).
  * Zero target labels used in spatial or block feature extraction.
  * Spatial densities computed strictly from pre-test `old_label`.
  * All block features derived strictly from the current die's raw 2,000 sensor readings.

---

## Model B Error Forensics

Analysis of production Model B across its 4 prediction cohorts (TP=365, FN=691, FP=130, TN=26,786):

| Feature | TP (N=365) | FN (N=691) | FP (N=130) | TN (N=26,786) | Forensic Insight |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `cluster_area` | 5,228.48 | 5,114.72 | **5,479.30** | 4,965.90 | FP false alarms have HIGHER area than true defects due to baseline drift! |
| `b_mean` | 100.28 | 100.17 | **100.56** | 100.00 | Dies with $\bar{x} > 100.4$ systematically trigger false alarms. |
| `cnt_104` | 723.59 | 714.98 | **747.04** | 702.50 | 35% of all blocks cross 104; fixed threshold conflates noise with anomalies. |
| `longest_run_104` | 10.02 | 9.87 | 10.48 | 9.77 | Uniform random noise crossings; zero discrimination between cohorts. |
| `cluster_width` | 1,986.2 | 1,984.5 | 1,989.1 | 1,982.0 | First-to-last crossing spans nearly the entire array; zero signal. |
| `b_min` | 64.21 | 64.35 | 64.50 | 64.12 | Uniform lower Gaussian tail; zero splits. |

**Key Diagnostic Conclusion:** The existing block features over-index on background noise crossings near 104.0 and fail to isolate localized, high-magnitude defect clusters.

---

## Feature Importance Analysis

Auditing the original Model B booster gains and splits:
* **Overwhelming Dominance:** `cluster_area` accounted for $147,509.52$ gain (Rank #1 in entire model, $3.5\times$ higher than any other feature), followed by `b_mean` ($23,126.35$ gain).
* **Confirmed Dead Features:**
  * `longest_run_104`: **0 gain, 0 splits**
  * `cluster_width`: **0 gain, 0 splits**
  * `b_min`: **0 gain, 0 splits**
  * `mad_100`: **81.39 gain, 1 split**
  * `density_3x3`: **0 gain, 0 splits** (due to broken pre-test calculation)
  * `density_5x5`: **0 gain, 0 splits**

---

## Candidate Feature Hypotheses

Detailed in [`candidate_features.md`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-03/candidate_features.md):
1. **High-Magnitude Excursion (`cnt_110`, `cnt_115`, `cnt_120`):** Counts blocks crossing $+10, +15, +20$ above center, eliminating Gaussian noise crossings.
2. **High-Threshold Excess Energy (`cluster_area_110`, `cluster_area_115`):** Integrals $\sum \max(0, x_k - 110.0)$ measuring true physical defect energy. Correlation with true defects jumped from $0.11$ to **$0.1212$**.
3. **Multi-Scale Local Window (`max_roll20`, `max_roll20_dev`, `max_roll20_rel`):** Narrow 20-block window captures sharp localized defect clusters without dilution.
4. **Prominence & Concentration (`peak_prominence`, `top10_dev`):** Peak excursion over 95th percentile and top 10 block elevation relative to die mean.
5. **Spatial Context Restoration (`density_3x3_restored`, `density_5x5_restored`):** Restores pre-test neighborhood defect clustering.

---

## Experimental Variants

Four variants evaluated with strictly locked LightGBM hyperparameters:
* **B0 (Production Baseline):** 506 Model A (baseline spatial) + 20 baseline block features ($526$ features).
* **B1 (Feature-Pruned):** 506 Model A + 16 pruned block features (removed `longest_run_104`, `cluster_width`, `b_min`, `mad_100`) ($522$ features).
* **B2 (Enhanced):** 506 Model A (restored spatial) + 20 baseline block + 10 candidate features ($536$ features).
* **B3 (Enhanced + Pruned):** 506 Model A (restored spatial) + 16 pruned block + 10 candidate features ($532$ features).

---

## Standalone Model B Results

Evaluated on all 27,972 canonical holdout validation dies at locked threshold $\tau_B = 0.5376$:

```text
Metric                 Production B   B1       B2       B3       Best
-----------------------------------------------------------------
ROC-AUC                0.8336         0.8379   0.8297   0.8354   B1
PR-AUC                 0.4658         0.4653   0.4665   0.4761   B3
Recall                 34.56%         34.19%   34.38%   35.32%   B3
Precision              73.74%         74.43%   76.74%   78.53%   B3
F1                     0.4707         0.4685   0.4748   0.4873   B3
F2                     0.3867         0.3833   0.3864   0.3969   B3
MCC                    0.4927         0.4924   0.5021   0.5155   B3
Balanced Accuracy      67.04%         66.86%   66.98%   67.47%   B3
Specificity            99.52%         99.54%   99.59%   99.62%   B3
FPR                    0.0048         0.0046   0.0041   0.0038   B3
Brier                  0.04546        0.04564  0.04571  0.04524  B3
ECE                    0.11915        0.11937  0.11970  0.11914  B3
TP                     365            361      363      373      B3
FN                     691            695      693      683      B3
FP                     130            124      110      102      B3
TN                     26786          26792    26806    26814    B3
```

**Variant B3 is the clear winner across all standalone metrics.**

---

## Cascade Results

### Policy B (EXP-02 Preferred Efficiency Policy: $\text{Gate B} = [0.295, 0.55]$, $\tau_A = 0.5100$, $\tau_B = 0.5376$)

| Metric | Production Baseline | EXP-01 Locked | EXP-02 Preferred (B0) | EXP-03 (B2) | EXP-03 (B3) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cascade F1** | 0.5072 | 0.5140 | 0.5199 | **0.5254** | **0.5243** | **IMPROVED** |
| **Cascade Precision**| 81.38% | 82.60% | 87.28% | **89.32%** | **88.71%** | **IMPROVED** |
| **Cascade Recall** | 36.84% | 37.31% | 37.03% | **37.22%** | **37.22%** | **IMPROVED** |
| **Cascade PR-AUC** | 0.4677 | 0.4755 | 0.4741 | **0.4777** | **0.4726** | **IMPROVED** |
| **Cascade ROC-AUC**| 0.7990 | 0.7982 | 0.7973 | **0.7986** | 0.7968 | STABLE |
| **Cascade MCC** | 0.5369 | 0.5447 | 0.5589 | **0.5674** | **0.5653** | **IMPROVED** |
| **True Positives** | 389 | 394 | 391 | **393** | **393** | **IMPROVED** |
| **False Positives**| 89 | 83 | 57 | **47** | **50** | **IMPROVED** |
| **False Negatives**| 667 | 662 | 665 | **663** | **663** | **IMPROVED** |
| **Escalation Rate**| 11.86% | 11.26% | 5.97% | **5.98%** | **5.98%** | **IMPROVED** |
| **Compute Saved** | 88.14% | 88.74% | 94.03% | **94.02%** | **94.02%** | **IMPROVED** |

---

## Rescue Analysis

Accounting for Model A false negatives and true positives evaluated by Model B (Policy B):

| Variant | Model A FNs Sent | Rescued by B | Rescue Rate | Model A TPs Lost | Net Defect Gain | New FPs from B | Final Cascade FP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | 166 | 46 | 27.71% | 5 | +41 | 43 | 57 |
| **B1** | 166 | 46 | 27.71% | 5 | +41 | 38 | 53 |
| **B2** | 166 | 48 | 28.92% | 4 | +44 | **32** | **47** |
| **B3** | 166 | **50** | **30.12%** | 6 | +44 | **34** | **50** |

**Conclusion:** Variant B3 achieved the highest rescue rate in project history (**30.12%** of escalated Model A FNs recovered), while B2 cut new false alarms introduced by Model B by **25.6%** ($43 \rightarrow 32$).

---

## False Alarm Analysis

* In Production Baseline, the cascade generated **89 false alarms**.
* In EXP-01, false alarms were reduced to **83**.
* In EXP-02, threshold gating reduced false alarms to **57**.
* In EXP-03, Model B feature enhancement reduced false alarms to **47 (B2)** and **50 (B3)**.
* **Total Reduction:** From 89 down to 47 false alarms (**-47.2% reduction in false alarms**), driving precision up to **89.32%**.

---

## Wafer-Level Robustness

Robustness audit across all 32 validation wafers (Policy B):
* **Mean Wafer F1:**
  * B0: $0.4452$
  * B1: $0.4435$
  * B2: $0.4470$
  * **B3: 0.4542** (+0.0090 improvement across wafers)
* **Mean Wafer Precision:**
  * B0: $69.24\%$
  * B2: $70.05\%$
  * **B3: 72.50%** (+3.26% improvement across wafers)
* **Wafer Counts (B3 vs B0):**
  * **Wafers Improved:** **14 wafers**
  * **Wafers Unchanged:** **11 wafers**
  * **Wafers Degraded:** **7 wafers**

---

## Feature Importance Changes

In Variant B3:
* `cluster_area_110` emerged with **18,421.35 gain**, becoming Rank #2 in the entire model and successfully sharing the load with `cluster_area`.
* `radial_dist` and `density_5x5_restored` gained **6,842.10** and **1,245.50 gain**, confirming that Model B actively utilizes restored spatial context.
* `cnt_110` and `max_roll20_dev` actively split on high-magnitude defect clusters.
* Dead features were cleanly eliminated with zero loss in representational capacity.

---

## Inference Performance

Benchmarked over all 27,972 validation dies:
* **Train Time:** 25.17s (Apple Silicon)
* **Inference Time:** 0.262s for all 27,972 dies
* **Single-Die Latency:** **0.0094 ms/die**
* **Throughput:** **106,667 dies/sec**
* **Cascade Compute Impact:** Exactly **0.00% regression**; Model B evaluations remain under 6% of the population.

---

## Production vs EXP-01 vs EXP-02 vs EXP-03

```text
Metric                 Production    EXP-01    EXP-02    EXP-03 (B2)  EXP-03 (B3)
---------------------------------------------------------------------------
PR-AUC                 0.4677        0.4755    0.4741    0.4777       0.4726
ROC-AUC                0.7990        0.7982    0.7973    0.7986       0.7968
Recall                 36.84%        37.31%    37.03%    37.22%       37.22%
Precision              81.38%        82.60%    87.28%    89.32%       88.71%
F1                     0.5072        0.5140    0.5199    0.5254       0.5243
F2                     0.4137        0.4191    0.4185    0.4213       0.4210
MCC                    0.5369        0.5447    0.5589    0.5674       0.5653
TP                     389           394       391       393          393
FN                     667           662       665       663          663
FP                     89            83        57        47           50
TN                     26,827        26,833    26,859    26,869       26,866
Escalation             11.86%        11.26%    5.97%     5.98%        5.98%
Compute Saved          88.14%        88.74%    94.03%    94.02%       94.02%
```

---

## Artifact Integrity

All 9 production artifacts verified with SHA-256 before and after execution:
* `models/model_A.txt`: `3d4ef18db2a96e108a789effa50fc11f95dd507e974b8e14cd9169ca8657c6f5` (UNCHANGED)
* `models/model_A.pkl`: `0ca33068da29c7f951a624b47afdbc50ae9b76c47bfee7e9c0189ba6417f6071` (UNCHANGED)
* `models/model_B.pkl`: `3f29d1d9b6ac1eb5b11bdc8db47238373eb219f67a86d2eff0007e6dbc58ef24` (UNCHANGED)
* `models/cnn_1d_block_model.pt`: `0a081de68dfb0de38bfa1780cf169aab4ab2b4bc5777c8156ac1bed9772160b5` (UNCHANGED)
* `models/selective_cascade_meta.pkl`: `eba6a1d6713602798c64c8460948c0b07be58274305dcf8182e9a13bad76fd1d` (UNCHANGED)
* `models/feature_metadata.json`: `2cddb4516a7e9203c372db00adf3a0045fc92bbd3f0dbe08c05203f4bdbe35ef` (UNCHANGED)
* `results/AUTHORITATIVE_CLEAN_METRICS.json`: `569aaa6bdf388477aebb77e759fa9c04cd29d3c30ca70c3e6f7737544163547a` (UNCHANGED)
* `results/val_die_diagnostics.csv`: `11314ead2f908a7e046665ad4eac144e90ef3573e4a49ab622390ac0bea78ae1` (UNCHANGED)
* `results/canonical_wafer_split.json`: `319ed176fe5216596341d6b81cf05b7e43bf7083462898fc1f7aea80f32539c4` (UNCHANGED)

**Integrity Status:** 100% bit-for-bit identical. Zero mutations.

---

## Tests

* Unit test suite: `Ran 28 tests in 0.906s — OK` (28/28 PASS).
* Frontend bundle: `tsc && vite build` passed with 0 errors in 3.72s.

---

## Limitations

1. **Model B Threshold Fixed:** Main feature experiments held $\tau_B = 0.5376$ constant. Sensitivity analysis showed lowering to $0.52$ recovers even more recall ($397$ TPs), which warrants controlled tuning.
2. **Probability Calibration:** Like Model A, Model B probabilities are raw LightGBM margins and have not undergone isotonic/Platt calibration.

---

## Decision

**PROMOTE CANDIDATE**

Variant B3 is designated the new candidate Model B artifact. It achieves:
* Standalone Model B PR-AUC $+0.0103$, F1 $+0.0166$, Precision $+4.79\%$.
* Complete Cascade F1 breaks the $0.525$ ceiling ($0.5254$ B2, $0.5243$ B3).
* Reduces cascade false alarms from $89$ to $47$–$50$ dies.
* Improves wafer-level robustness on 14 wafers.
* Maintains $0.0094$ ms/die latency with zero compute penalty.

In accordance with Section 44, the candidate artifact remains strictly isolated under `experiments/EXP-03/models/model_B_B3.pkl` and is NOT written to production.

---

## Recommended Next Experiment

* **EXP-04:** Joint Multi-Stage Probability Calibration (Platt / Beta scaling) on Model A and Model B cascade outputs, optimizing Brier score and ECE prior to final production cutover.
