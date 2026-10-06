# EXP-04: Feature Provenance and Spatial Safety Audit

## Overview
This document certifies the feature provenance, mathematical formulations, and strict data leakage controls for all features utilized by the frozen **Model A** and **Model B B2** pipelines during **EXP-04** independent holdout evaluation on `input/test.csv`.

---

## 1. Feature Representation Summary

| Feature Family | Count | Primary Models | Formula / Source | Available at Inference? | Target / Label Dependence |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **Parametric Electrical** | 500 | Model A, Model B B2 | `feature_1` to `feature_500` (Automated Test Equipment electrical measurements) | **YES** | **NONE (0%)** |
| **Spatial Wafer Context** | 6 | Model A, Model B B2 | `radial_dist`, `edge_dist`, `die_row`, `die_col`, `density_3x3_restored`, `density_5x5_restored` | **YES** | **NONE (0%)** |
| **Baseline Sub-Die Block** | 20 | Model B B2 | Statistical aggregations from raw 2,000 sub-die sequence floats | **YES** | **NONE (0%)** |
| **Frozen Block Candidates**| 10 | Model B B2 | High-resolution threshold counters, cluster lengths, rolling windows | **YES** | **NONE (0%)** |
| **Total Features** | **536** | Model B B2 (506 Model A) | Complete Multi-Resolution Feature Space | **YES** | **100% LEAK-FREE** |

---

## 2. Spatial Feature Safety Certification

### Restored Spatial Density Features (`density_3x3_restored`, `density_5x5_restored`)
- **Source Columns**: `wafer_id`, `die_row`, `die_col`, and `old_label`.
- **Target Independence**: These features **never** access `label` (post-packaging burn-in failure target). They utilize strictly `old_label` (pre-assembly wafer-probe gross functional rejects).
- **Physical Rationale**: At wafer saw and packaging, the factory has already executed wafer probe testing (`old_label`). The locations of pre-test failed dies on the wafer map are physically known and available before packaging. A die surrounded by gross pre-test rejects has an elevated physical likelihood of latent defects due to localized fabrication contamination or crystal dislocations.
- **Mathematical Formula**:
  $$\text{density\_3x3}(r, c) = \sum_{i=-1}^{1} \sum_{j=-1}^{1} \mathbb{I}[(r+i, c+j) \in \text{Wafer}] \cdot \text{old\_label}(r+i, c+j)$$
  $$\text{density\_5x5}(r, c) = \sum_{i=-2}^{2} \sum_{j=-2}^{2} \mathbb{I}[(r+i, c+j) \in \text{Wafer}] \cdot \text{old\_label}(r+i, c+j)$$
- **Zero Future / Neighbor Contamination**: Computed strictly per-wafer using the wafer map of the current wafer being inspected. No cross-wafer statistics, no test-set aggregation, and zero access to post-test labels.

---

## 3. Frozen Block Feature Safety Certification

All 10 frozen engineered block features are computed strictly on each die's own 2,000-block time series vector $S = [s_1, s_2, \dots, s_{2000}]$:
1. `cnt_110`: $\sum_{i=1}^{2000} \mathbb{I}[s_i > 110.0]$
2. `cnt_115`: $\sum_{i=1}^{2000} \mathbb{I}[s_i > 115.0]$
3. `cnt_120`: $\sum_{i=1}^{2000} \mathbb{I}[s_i > 120.0]$
4. `cluster_area_110`: Total length of contiguous block runs where $s_i > 110.0$.
5. `cluster_area_115`: Total length of contiguous block runs where $s_i > 115.0$.
6. `max_roll20`: $\max_{k} \left( \frac{1}{20} \sum_{j=0}^{19} s_{k+j} \right)$
7. `max_roll20_dev`: $\max\_roll20 - \bar{S}$ (where $\bar{S}$ is the die's own mean).
8. `max_roll20_rel`: $\max\_roll20 / (\bar{S} + 10^{-6})$
9. `peak_prominence`: $\max(S) - \text{Percentile}_{95}(S)$
10. `top10_dev`: $\text{Mean}(\text{Top10}(S)) - \bar{S}$

Each calculation is completely die-internal and deterministic, with zero inter-die or label dependencies.
