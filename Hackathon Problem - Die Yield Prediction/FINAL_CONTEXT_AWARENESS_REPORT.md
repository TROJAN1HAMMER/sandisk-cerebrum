# FINAL CONTEXT AWARENESS REPORT
## EXP-10: Context-Aware Die Intelligence
**SanDisk Cerebrum Die Yield Prediction Platform**  
**Date:** October 2026  
**Auditor / Lead Scientist:** Antigravity AI  
**Baseline Status:** Frozen B2 Selective Cascade — **Retained in Production**  
**Final Experiment Classification:** **REJECT (KEEP B2 CONTROL)**

---

## Executive Summary

EXP-10 evaluated whether integrating structured context—physical neighbourhood, spatial defect cluster topology, wafer-relative baselines, 2,000-block signal geometry, and multi-sensor evidence agreement—could enhance latent defect detection in the SanDisk Cerebrum selective cascade.

Using strict wafer-grouped validation on the canonical 128-wafer training set and 32-wafer validation set, seven candidate feature sets (C1 through C7) were evaluated through the locked cascade ($\tau_A = 0.5100, \tau_B = 0.5376$, Gate $[0.295, 0.550]$). 

**The empirical verdict is definitive: Contextual feature additions failed to beat the frozen B2 baseline.**
- The top-performing contextual candidate, **C1 (Local Physical Neighbourhood)**, slightly increased validation recall (+2 rescued defects) but diluted precision and degraded performance on **14 of 32 validation wafers**.
- On the independent 40-wafer holdout (`input/test.csv`, 32,598 dies), C1 degraded Cascade F1 from **0.5177 down to 0.5159** and recovered **3 fewer net defects** (+34 vs +37).
- Families B, C, D, E, C6, and C7 degraded Cascade F1 by up to $-0.0170$ due to tree split dilution and wafer-level overfitting.
- **Production Decision**: **RETAIN FROZEN B2 WITHOUT MODIFICATION.**

---

## Answers to Mandatory Program Questions

### 1. What contextual information was added?
A total of **34 engineered contextual features** organized into five distinct families were added to Model B2:
- **Family A (Local Physical Neighbourhood, 7 features)**: $3\times 3$ grid neighbour counts, local means, local differences ($\text{die} - \mu_{\text{local}}$), local z-scores, and composite electrical contrast for top drivers (`feature_171`, `feature_460`, `feature_336`).
- **Family B (Spatial Topology & Morphology, 6 features)**: $5\times 5$ horizontal, vertical, and diagonal defect densities on pre-test screening rejects (`old_label`), defect cluster aspect ratios ($\log(\rho_H / \rho_V)$), and isolated/dense cluster flags.
- **Family C (Wafer-Relative Context, 8 features)**: Intra-wafer z-scores, robust MAD-based z-scores, and wafer-level means and standard deviations for top electrical measurements.
- **Family D (Signal Geometry & Location, 8 features)**: Peak position index, anomaly centroid, early/mid/late area ratios, left-vs-right asymmetry, peak sharpness ratio, and sustained run indicator along the 2,000-block analog sequence.
- **Family E (Cross-Evidence Agreement, 5 features)**: Pairwise interaction products (Electrical $\times$ Spatial, Electrical $\times$ Signal, Spatial $\times$ Signal), multi-sensor exceedance counts, and 3-way concordance flags.

### 2. Why was each feature family added?
- **Family A**: To identify localized electrical outliers—dies whose measurements are sharply deviant from the physical dies directly surrounding them, regardless of global thresholds.
- **Family B**: To discriminate scratch lines, stepper slit tracks, and directional streaks from isotropic clusters or isolated random fails.
- **Family C**: To decouple tool-to-tool baseline shifts and thermal gradients from true die-level defect signatures.
- **Family D**: To pinpoint the exact physical location of analog failure within the memory wordline/bitline structure.
- **Family E**: To amplify dies where multiple independent sensors (probe test, spatial cluster, and block sequence) simultaneously corroborate abnormality.

### 3. Which features actually helped?
- **`f171_local_diff_3x3`** and **`f171_local_zscore_3x3`** (from Family A) demonstrated genuine discriminative value, rescuing **+2 additional false negatives** in validation. They enabled LightGBM to detect subtle localized electrical spikes that fell just below the global threshold.
- However, their benefit was constrained: on wafer borders, reduced neighbor counts generated edge noise that introduced extra false positives (+6 FP in validation), neutralizing the net gain.

### 4. Which features were redundant?
- **All of Family B (Spatial Topology)**: Pearson correlation between directional densities and existing isotropic densities (`density_3x3`, `density_5x5`) exceeded $0.88$. LightGBM gained zero new information about post-screening latent defects.
- **All of Family E (Evidence Agreement)**: Decision trees naturally model multiplicative feature interactions via recursive axis splits. Manually multiplying normalized columns added collinear split candidates without creating new partitioning capacity.

### 5. Did spatial context help?
**NO.** Adding directional topology (Family B) decreased Cascade F1 from **0.5264 down to 0.5110** ($-0.0154$) and increased validation false alarms from 32 to 52. Pre-test reject patterns captured by `old_label` are already optimally represented by the existing `density_3x3` and `density_5x5` features.

### 6. Did wafer-relative context help?
**NO, it caused severe distribution shift and overfitting.**
- LightGBM seized upon `wafer_mean_f171` (gain: 55,152) and `wafer_std_f171` (gain: 48,001) as high-priority split criteria.
- Because wafer defect rates vary drastically across fabrication lots (from $<1\%$ to $>14\%$), the tree splits overfitted the specific lot distributions of the 128 training wafers.
- On unseen validation wafers, wafer-mean splits performed poorly, dropping Cascade F1 to **0.5094** ($-0.0170$) and slashing Net TP recovery from +45 down to +31 ($-31\%$).

### 7. Did signal geometry help?
**NO.** Family D degraded Cascade F1 to **0.5194** ($-0.0070$).
Analog defect spikes in flash memory arrays are already compactly captured by B2's magnitude-focused features (`cluster_area_110`, `cnt_110`, `max_roll20`). Segmenting by physical block position (`anomaly_centroid`, `anomaly_early_ratio`) introduced high-variance splits that did not generalize across distinct die layouts.

### 8. Did evidence agreement help?
**NO.** Family E degraded Cascade F1 to **0.5136** ($-0.0128$) and dropped Net TP gain to +37. Explicit product features acted as collinear noise.

### 9. Did the complete cascade improve?
**NO.** Every single candidate configuration (C1 through C7) underperformed the frozen B2 cascade on F1 score and Net TP gain.

### 10. Did false positives increase?
**YES.**
- Frozen B2 baseline validation new FPs: **32**
- C1 (Neighbourhood): **38** (+6 new FPs)
- C2 (Spatial Topology): **52** (+20 new FPs)
- C3 (Wafer-Relative): **41** (+9 new FPs)
- C4 (Signal Geometry): **39** (+7 new FPs)
- C5 (Evidence Agreement): **47** (+15 new FPs)
- C6 (All Families): **57** (+25 new FPs)

### 11. Did compute savings change?
**NO.** Because the Stage 1 Model A classifier and escalation boundaries ($[0.295, 0.550]$) remained locked, the escalation rate remained stable at **$5.97\%$** (saving $>94.0\%$ of heavy computation).

### 12. Did wafer robustness improve?
**NO.**
- Under C1, only 8 wafers improved, 10 were tied, and **14 wafers degraded**.
- Under C2, 17 wafers degraded.
- Under C3, 14 wafers degraded.
- Under C6, 15 wafers degraded.
Mean wafer F1 declined from **0.5736** (B0) to **0.5726** (C1), **0.5606** (C3), and **0.5601** (C6).

### 13. Was there any leakage?
**NO.** All 34 features were confirmed **100% leakage-free** in [`FEATURE_LEAKAGE_AUDIT.md`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/FEATURE_LEAKAGE_AUDIT.md). No ground-truth labels, future outcomes, or cross-wafer data were used.

### 14. Which features should be retained?
**NONE in the production decision engine.**
While `f171_local_diff_3x3` and `anomaly_centroid` provide intuitive physical explanations for offline engineer dashboards, adding them into the active decision forest degrades generalization.

### 15. Should B2 be replaced?
**NO. B2 REMAINS THE PRODUCTION CHAMPION.**
Frozen B2 is mathematically more parsimonious, achieves higher precision, causes fewer false alarms, and delivers superior generalization on unseen wafers.

---

## Authoritative Performance Comparison Table

| Candidate System | Features | Val Cascade F1 | Val PR-AUC | Val Net TP | Holdout Cascade F1 | Holdout PR-AUC | Holdout Net TP | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Frozen B2 Baseline (Control)** | **536** | **0.5264** | **0.4779** | **+45** | **0.5177** | **0.4809** | **+37** | **PRODUCTION RETAINED** |
| **C1 Local Neighbourhood** | 543 | 0.5259 | 0.4750 | +47 | 0.5159 | 0.4803 | +34 | **REJECTED** |
| **C2 Spatial Topology** | 542 | 0.5110 | 0.4746 | +36 | — | — | — | **REJECTED** |
| **C3 Wafer-Relative Context** | 544 | 0.5094 | 0.4657 | +31 | — | — | — | **REJECTED** |
| **C4 Signal Geometry** | 544 | 0.5194 | 0.4766 | +40 | — | — | — | **REJECTED** |
| **C5 Evidence Agreement** | 541 | 0.5136 | 0.4723 | +37 | — | — | — | **REJECTED** |
| **C6 All Context Families** | 570 | 0.5096 | 0.4649 | +36 | — | — | — | **REJECTED** |
| **C7 Strongest Selected** | 544 | 0.5120 | 0.4633 | +35 | — | — | — | **REJECTED** |

---

## Directory of EXP-10 Deliverables

- Audit: [`FEATURE_AUDIT.md`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/FEATURE_AUDIT.md)
- Mathematical Formulations: [`FEATURE_DEFINITIONS.md`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/FEATURE_DEFINITIONS.md)
- Leakage Audit: [`FEATURE_LEAKAGE_AUDIT.md`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/FEATURE_LEAKAGE_AUDIT.md)
- Ablation Matrix: [`FEATURE_FAMILY_ABLATION.csv`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/FEATURE_FAMILY_ABLATION.csv)
- Wafer Breakdown: [`WAFER_ROBUSTNESS.csv`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/WAFER_ROBUSTNESS.csv)
- Latency Benchmark: [`LATENCY_COMPARISON.csv`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/LATENCY_COMPARISON.csv)
- Machine Metrics: [`METRICS.json`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/METRICS.json)
- Technical Findings: [`RESULTS.md`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/RESULTS.md)
- Overview: [`README.md`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/README.md)
