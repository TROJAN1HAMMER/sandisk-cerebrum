# EXP-10: Feature Leakage Audit

**Experiment:** EXP-10 Context-Aware Die Intelligence  
**Platform:** SanDisk Cerebrum Die Yield Prediction  
**Status:** Verification & Compliance Audit  
**Date:** October 2026  

---

## Executive Audit Summary

Every candidate feature proposed for EXP-10 was audited against the six non-negotiable data leakage criteria:
1. Does it use the target label (`label`)?
2. Does it use neighbouring target labels?
3. Does it use post-test outcomes?
4. Does it use future information unavailable at Sort-1?
5. Does it use information unavailable during real inference?
6. Does wafer-level normalization leak across validation or holdout partitions?

**AUDIT RESULT: 100% LEAKAGE-FREE COMPLIANCE.**  
All 34 proposed features across Families A through E use strictly pre-test screening data (`old_label`), raw electrical parametric measurements, physical wafer grid coordinates, or 2,000-block analog sequence readings available at the moment a die is tested at probe station Sort-1.

---

## Detailed Feature Leakage Inventory

### Family A: Local Physical Neighbourhood

| Feature Name | Primary Source | Calculation | Inference Availability | Leakage Status | Causal Justification |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `valid_neighbors_3x3` | Wafer die grid coordinates | Count of active dies in $3\times 3$ window on same wafer | YES | **PASS (NO LEAKAGE)** | Grid coordinates are fixed physical design properties. |
| `f171_local_mean_3x3` | `feature_171` | Uniform filter over $3\times 3$ grid divided by valid mask | YES | **PASS (NO LEAKAGE)** | Uses only input measurement of neighbouring dies on same wafer. No target label used. |
| `f171_local_diff_3x3` | `feature_171` | Die value minus local mean | YES | **PASS (NO LEAKAGE)** | Computed purely from electrical input space. |
| `f171_local_zscore_3x3` | `feature_171` | $(x - \mu_{3\times 3}) / (\sigma_{3\times 3} + \epsilon)$ | YES | **PASS (NO LEAKAGE)** | Local sample variance on input features only. |
| `f460_local_diff_3x3` | `feature_460` | Die value minus local mean | YES | **PASS (NO LEAKAGE)** | Input measurement difference only. |
| `f336_local_diff_3x3` | `feature_336` | Die value minus local mean | YES | **PASS (NO LEAKAGE)** | Input measurement difference only. |
| `local_electrical_contrast`| `feature_171, 460, 336` | Euclidean norm of local differences | YES | **PASS (NO LEAKAGE)** | Non-linear composite of valid input differences. |

---

### Family B: Spatial Topology & Morphology

| Feature Name | Primary Source | Calculation | Inference Availability | Leakage Status | Causal Justification |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `h_defect_density_5x5` | `old_label`, die grid | 1D horizontal window sum of pre-test rejects | YES | **PASS (NO LEAKAGE)** | `old_label` is the screening test result obtained *before* electrical probe test. Completely known at inference time. |
| `v_defect_density_5x5` | `old_label`, die grid | 1D vertical window sum of pre-test rejects | YES | **PASS (NO LEAKAGE)** | Pre-test screening outcome only. Zero future target information. |
| `diag_defect_density_5x5`| `old_label`, die grid | Diagonal window sum of pre-test rejects | YES | **PASS (NO LEAKAGE)** | Pre-test screening outcome only. |
| `cluster_aspect_ratio` | `old_label` | $\log((\rho_H + 0.01) / (\rho_V + 0.01))$ | YES | **PASS (NO LEAKAGE)** | Mathematical ratio of known screening spatial densities. |
| `isolated_defect_flag` | `old_label` | $\mathbb{I}(\text{density}_{3\times 3}=0 \land \text{density}_{5\times 5}>0)$ | YES | **PASS (NO LEAKAGE)** | Boolean combination of existing valid spatial features. |
| `dense_cluster_flag` | `old_label` | $\mathbb{I}(\text{density}_{5\times 5} \ge 0.40)$ | YES | **PASS (NO LEAKAGE)** | Thresholding of existing valid spatial density. |

---

### Family C: Wafer-Relative Context

| Feature Name | Primary Source | Calculation | Inference Availability | Leakage Status | Causal Justification |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `f171_wafer_zscore` | `feature_171`, `wafer_id` | $(x - \mu_W) / (\sigma_W + \epsilon)$ | YES | **PASS (NO LEAKAGE)** | Evaluated strictly *within* the single wafer. Zero cross-wafer pooling. During production inference, a wafer lot is tested together; computing intra-wafer mean/std violates zero causal boundaries. |
| `f171_wafer_robust_z` | `feature_171`, `wafer_id` | $(x - M_W) / (1.4826 \cdot \text{MAD}_W + \epsilon)$ | YES | **PASS (NO LEAKAGE)** | Median and MAD computed strictly within wafer. |
| `f460_wafer_zscore` | `feature_460`, `wafer_id` | $(x - \mu_W) / (\sigma_W + \epsilon)$ | YES | **PASS (NO LEAKAGE)** | Intra-wafer normalization only. |
| `f336_wafer_zscore` | `feature_336`, `wafer_id` | $(x - \mu_W) / (\sigma_W + \epsilon)$ | YES | **PASS (NO LEAKAGE)** | Intra-wafer normalization only. |
| `f33_wafer_zscore` | `feature_33`, `wafer_id` | $(x - \mu_W) / (\sigma_W + \epsilon)$ | YES | **PASS (NO LEAKAGE)** | Intra-wafer normalization only. |
| `f109_wafer_zscore` | `feature_109`, `wafer_id` | $(x - \mu_W) / (\sigma_W + \epsilon)$ | YES | **PASS (NO LEAKAGE)** | Intra-wafer normalization only. |
| `wafer_mean_f171` | `feature_171`, `wafer_id` | $\frac{1}{|W|} \sum_{i \in W} f_{171}(i)$ | YES | **PASS (NO LEAKAGE)** | Mean of input features on this wafer. |
| `wafer_std_f171` | `feature_171`, `wafer_id` | Standard deviation of $f_{171}$ on this wafer | YES | **PASS (NO LEAKAGE)** | Standard deviation of input features on this wafer. |

---

### Family D: Signal Geometry & Location

| Feature Name | Primary Source | Calculation | Inference Availability | Leakage Status | Causal Justification |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `anomaly_peak_pos` | `block_readings` (2,000) | $\arg\max(b) / 2000.0$ | YES | **PASS (NO LEAKAGE)** | Die's own analog sequence. No cross-die data used. |
| `anomaly_centroid` | `block_readings` (2,000) | Center of mass above baseline $100$ | YES | **PASS (NO LEAKAGE)** | Die's own analog sequence. |
| `anomaly_early_ratio` | `block_readings` (2,000) | Area in blocks $0 \dots 666$ / total area | YES | **PASS (NO LEAKAGE)** | Die's own analog sequence. |
| `anomaly_mid_ratio` | `block_readings` (2,000) | Area in blocks $667 \dots 1333$ / total area | YES | **PASS (NO LEAKAGE)** | Die's own analog sequence. |
| `anomaly_late_ratio` | `block_readings` (2,000) | Area in blocks $1334 \dots 1999$ / total area | YES | **PASS (NO LEAKAGE)** | Die's own analog sequence. |
| `anomaly_asymmetry` | `block_readings` (2,000) | $(A_{\text{late}} - A_{\text{early}}) / (A_{\text{total}} + \epsilon)$ | YES | **PASS (NO LEAKAGE)** | Die's own analog sequence. |
| `sharp_peak_ratio` | `block_readings` (2,000) | Peak window area / total area | YES | **PASS (NO LEAKAGE)** | Die's own analog sequence. |
| `sustained_anomaly_flag`| `block_readings` (2,000) | $\mathbb{I}(\text{longest\_run\_104} \ge 50)$ | YES | **PASS (NO LEAKAGE)** | Die's own analog sequence. |

---

### Family E: Cross-Evidence Agreement & Concordance

| Feature Name | Primary Source | Calculation | Inference Availability | Leakage Status | Causal Justification |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `elec_spatial_agreement` | Families A, B, C | $E \times S$ | YES | **PASS (NO LEAKAGE)** | Product of leakage-free input features. |
| `elec_signal_agreement` | Families C, D | $E \times B$ | YES | **PASS (NO LEAKAGE)** | Product of leakage-free input features. |
| `spatial_signal_agreement`| Families B, D | $S \times B$ | YES | **PASS (NO LEAKAGE)** | Product of leakage-free input features. |
| `concordant_evidence_count`| Families B, C, D | Sum of binary sensor exceedances | YES | **PASS (NO LEAKAGE)** | Count of active leakage-free input signals. |
| `full_concordance_flag` | Families B, C, D | $\mathbb{I}(N_{\text{conc}} = 3)$ | YES | **PASS (NO LEAKAGE)** | Boolean combination of valid input exceedances. |

---

## Conclusion
Every feature in EXP-10 satisfies all regulatory and machine-learning integrity requirements. No downstream, future, target, or out-of-wafer information is used.
