# EXP-03 Candidate Feature Hypotheses

## Overview
Model B serves as the second-stage selective classifier, called only for dies in the uncertainty escalation band (e.g. $[0.295, 0.55]$ in EXP-02, or $[0.25, 0.55]$ in baseline). 
The goal of EXP-03 feature engineering is **hypothesis-driven refinement of block anomaly representation** to solve Model B's primary failure modes:
1. **False Negatives:** Subtle defect excursions that fail to cross the global $104.0$ threshold or produce small aggregate cluster areas.
2. **False Positives:** Healthy dies with slight global baseline drift ($\bar{x} > 100.4$) where random Gaussian noise crosses $104.0$ repeatedly, creating false cluster areas.

---

## Candidate Features Specification

### Family A: Higher-Threshold Energy (Filtering Confounding Baseline Noise)

| Feature Name | Mathematical Definition | Physical Interpretation | Expected Benefit | Potential Leakage Risk |
| :--- | :--- | :--- | :--- | :--- |
| `cnt_110` | $\sum_{k=1}^{2000} \mathbb{I}(x_k > 110.0)$ | Count of blocks exceeding $110.0$ ($+10.0$ / $\sim 1\sigma$ excursion). | Suppresses random background noise (where $35\%$ of blocks cross $104.0$); highlights genuine defect activity. | **Zero**: derived solely from raw sensor readings of the current die prior to test. |
| `cnt_115` | $\sum_{k=1}^{2000} \mathbb{I}(x_k > 115.0)$ | Count of blocks exceeding $115.0$ ($+15.0$ / $\sim 1.5\sigma$ excursion). | Isolates intermediate-to-high defect spikes that cannot be produced by Gaussian noise tails. | **Zero**: derived solely from raw sensor readings of the current die. |
| `cnt_120` | $\sum_{k=1}^{2000} \mathbb{I}(x_k > 120.0)$ | Count of blocks exceeding $120.0$ ($+20.0$ / $\sim 2\sigma$ excursion). | Pinpoints severe physical memory cell breakdown without sensitivity to baseline shift. | **Zero**: derived solely from raw sensor readings of the current die. |
| `cluster_area_110` | $\sum_{k=1}^{2000} \max(0, x_k - 110.0)$ | Excess integrated signal energy above $110.0$. | Replaces noisy `cluster_area` (at $104.0$) with clean anomaly mass, eliminating false alarms on baseline-shifted dies. | **Zero**: derived solely from raw sensor readings of the current die. |
| `cluster_area_115` | $\sum_{k=1}^{2000} \max(0, x_k - 115.0)$ | Excess integrated signal energy above $115.0$. | High-confidence defect energy integral. | **Zero**: derived solely from raw sensor readings of the current die. |

---

### Family B: Multi-Scale Local Morphology & Prominence

| Feature Name | Mathematical Definition | Physical Interpretation | Expected Benefit | Potential Leakage Risk |
| :--- | :--- | :--- | :--- | :--- |
| `max_roll20` | $\max_k \left( \frac{1}{20} \sum_{j=-9}^{10} x_{k+j} \right)$ | Maximum 20-block rolling window average. | Detects localized defect clusters that are diluted by the existing 50- and 100-block rolling filters. | **Zero**: deterministic spatial-sequential 1D filter on die's own sensor signal. |
| `max_roll20_dev` | $\text{max\_roll20} - 100.0$ | Deviation of peak 20-block window from nominal center ($100.0$). | Fast-path indicator of localized cluster elevation. | **Zero**: deterministic scalar comparison. |
| `max_roll20_rel` | $\text{max\_roll20} - \bar{x}_{\text{die}}$ | Peak 20-block window deviation relative to the die's own empirical mean. | Decouples local defect cluster from die-level baseline drift, preventing false alarms. | **Zero**: uses only within-die readings. |
| `peak_prominence` | $x_{\max} - x_{p95}$ | Difference between maximum reading and 95th percentile. | Distinguishes isolated/sharp anomaly spikes from diffuse high-variance noise. | **Zero**: quantile computation on current die readings. |
| `top10_dev` | $\left( \frac{1}{10} \sum_{j=1}^{10} x_{(2000-j+1)} \right) - \bar{x}_{\text{die}}$ | Average of top 10 highest readings relative to die mean. | Sensitive measure of concentrated anomaly magnitude. | **Zero**: within-die ranking. |

---

### Family C: Spatial Defect Density Restoration in Model B

| Feature Name | Mathematical Definition | Physical Interpretation | Expected Benefit | Potential Leakage Risk |
| :--- | :--- | :--- | :--- | :--- |
| `density_3x3_restored` | $\frac{1}{8} \sum_{\text{neighbors } 3\times 3} \text{old\_label}$ | Fraction of pre-test rejected dies in $3\times 3$ grid. | Replaces constant zero feature in production Model B with validated EXP-01 pre-test spatial clustering. | **Zero**: uses strictly pre-test `old_label` on wafer map; target `label` is completely isolated. |
| `density_5x5_restored` | $\frac{1}{24} \sum_{\text{neighbors } 5\times 5} \text{old\_label}$ | Fraction of pre-test rejected dies in $5\times 5$ grid. | Restores meso-scale neighborhood defect density. | **Zero**: uses strictly pre-test `old_label` on wafer map; target `label` is completely isolated. |

---

## Features Proposed for Pruning (Dead/Redundant Features)

From the production Model B feature gain audit:
1. `longest_run_104`: $0$ gain, $0$ splits (random noise runs cross 104 with length 9–11 uniformly across all dies).
2. `cluster_width`: $0$ gain, $0$ splits (first to last crossing of 104 spans almost the entire 2000 blocks for healthy dies with $\bar{x} > 100.2$).
3. `b_min`: $0$ gain, $0$ splits (lower noise tail has no defect predictive power).
4. `mad_100`: $81.39$ gain, $1$ split (redundant with `b_std` and `b_mean`).
