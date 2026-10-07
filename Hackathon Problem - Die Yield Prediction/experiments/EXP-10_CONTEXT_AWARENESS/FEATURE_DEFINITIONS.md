# EXP-10: Mathematical Feature Definitions

**Experiment:** EXP-10 Context-Aware Die Intelligence  
**Platform:** SanDisk Cerebrum Die Yield Prediction  
**Status:** Feature Specification Document  
**Date:** October 2026  

---

## Overview

EXP-10 introduces five distinct contextual feature families designed to provide local physical, spatial morphological, wafer-relative, signal-geometric, and multi-sensor concordance context to the B2 LightGBM model.

---

## Family A: Local Physical Neighbourhood (7 features)

Physical neighbours are defined on the die coordinate grid $(r, c)$ on the same wafer $W$. For a $3 \times 3$ window centered at die $(r_0, c_0)$, the local neighborhood set of valid dies is:
$$\mathcal{N}_3(r_0, c_0) = \{(r, c) \in W \mid |r - r_0| \le 1, |c - c_0| \le 1\}$$

1. **`valid_neighbors_3x3`**:
   $$N_{\text{valid}} = |\mathcal{N}_3(r_0, c_0)|$$
   *Range*: $1 \dots 9$. Identifies dies on the wafer edge or missing corners where neighbor support is restricted.

2. **`f171_local_mean_3x3`**:
   $$\mu_{3\times 3}(f_{171}) = \frac{1}{|\mathcal{N}_3|} \sum_{(r, c) \in \mathcal{N}_3} f_{171}(r, c)$$
   *Meaning*: Local baseline value for the primary electrical defect driver (`feature_171`).

3. **`f171_local_diff_3x3`**:
   $$\Delta_{3\times 3}(f_{171}) = f_{171}(r_0, c_0) - \mu_{3\times 3}(f_{171})$$
   *Meaning*: Die-level excess relative to surrounding neighbours. An isolated spike yields $\Delta \gg 0$.

4. **`f171_local_zscore_3x3`**:
   $$z_{3\times 3}(f_{171}) = \frac{f_{171}(r_0, c_0) - \mu_{3\times 3}(f_{171})}{\sigma_{3\times 3}(f_{171}) + \epsilon}$$
   where $\sigma_{3\times 3}$ is the sample standard deviation of $f_{171}$ across $\mathcal{N}_3$, and $\epsilon = 10^{-5}$.

5. **`f460_local_diff_3x3`**:
   $$\Delta_{3\times 3}(f_{460}) = f_{460}(r_0, c_0) - \mu_{3\times 3}(f_{460})$$
   *Meaning*: Local excess for secondary electrical driver (`feature_460`).

6. **`f336_local_diff_3x3`**:
   $$\Delta_{3\times 3}(f_{336}) = f_{336}(r_0, c_0) - \mu_{3\times 3}(f_{336})$$
   *Meaning*: Local excess for tertiary electrical driver (`feature_336`).

7. **`local_electrical_contrast`**:
   $$C_{\text{elec}} = \sqrt{|\Delta_{3\times 3}(f_{171})|^2 + |\Delta_{3\times 3}(f_{460})|^2 + |\Delta_{3\times 3}(f_{336})|^2}$$
   *Meaning*: Composite Euclidean magnitude of multi-channel electrical contrast against local neighbours.

---

## Family B: Spatial Topology & Morphology (6 features)

Evaluated on the $5 \times 5$ window centered at $(r_0, c_0)$ over the pre-test screening rejects map (`old_label` $\in \{0, 1\}$).

1. **`h_defect_density_5x5`**:
   Horizontal defect concentration across row $r_0$ within column range $[c_0 - 2, c_0 + 2]$:
   $$\rho_H = \frac{\sum_{c = c_0 - 2}^{c_0 + 2} \text{old\_label}(r_0, c)}{\sum_{c = c_0 - 2}^{c_0 + 2} \text{valid}(r_0, c) + \epsilon}$$
   *Meaning*: Detects horizontal scratch tracks produced by wafer handling end-effectors.

2. **`v_defect_density_5x5`**:
   Vertical defect concentration across column $c_0$ within row range $[r_0 - 2, r_0 + 2]$:
   $$\rho_V = \frac{\sum_{r = r_0 - 2}^{r_0 + 2} \text{old\_label}(r, c_0)}{\sum_{r = r_0 - 2}^{r_0 + 2} \text{valid}(r, c_0) + \epsilon}$$
   *Meaning*: Detects vertical track/slit defects from stepper scanning artifacts.

3. **`diag_defect_density_5x5`**:
   Defect concentration along the main and anti-diagonals in the $5 \times 5$ window:
   $$\rho_{\text{diag}} = \frac{\sum_{k = -2}^2 (\text{old\_label}(r_0+k, c_0+k) + \text{old\_label}(r_0+k, c_0-k))}{\sum_{k = -2}^2 (\text{valid}(r_0+k, c_0+k) + \text{valid}(r_0+k, c_0-k)) + \epsilon}$$

4. **`cluster_aspect_ratio`**:
   Elongation metric of defect concentration:
   $$\text{AR} = \log\left(\frac{\rho_H + 0.01}{\rho_V + 0.01}\right)$$
   *Meaning*: Positive values indicate horizontally elongated clusters; negative values indicate vertically elongated clusters; near zero indicates isotropic clusters.

5. **`isolated_defect_flag`**:
   $$I_{\text{iso}} = \mathbb{I}\left(\text{density}_{3\times 3} = 0 \text{ and } \text{density}_{5\times 5} > 0\right)$$
   *Meaning*: Distinguishes isolated border dies from clustered core defects.

6. **`dense_cluster_flag`**:
   $$I_{\text{dense}} = \mathbb{I}\left(\text{density}_{5\times 5} \ge 0.40\right)$$
   *Meaning*: Flags dies embedded within high-density defect epicenters.

---

## Family C: Wafer-Relative Context (8 features)

For each wafer $W$, sample statistics of electrical measurements are computed across all valid dies on that wafer:
$$\mu_W(f_k) = \frac{1}{|W|} \sum_{i \in W} f_k(i), \quad \sigma_W(f_k) = \sqrt{\frac{1}{|W|} \sum_{i \in W} (f_k(i) - \mu_W(f_k))^2}$$
$$M_W(f_k) = \text{median}_{i \in W}(f_k(i)), \quad \text{MAD}_W(f_k) = \text{median}_{i \in W}(|f_k(i) - M_W(f_k)|)$$

1. **`f171_wafer_zscore`**:
   $$z_{W}(f_{171}) = \frac{f_{171} - \mu_W(f_{171})}{\sigma_W(f_{171}) + 10^{-5}}$$
2. **`f171_wafer_robust_z`**:
   $$z_{W,\text{rob}}(f_{171}) = \frac{f_{171} - M_W(f_{171})}{\text{MAD}_W(f_{171}) \times 1.4826 + 10^{-5}}$$
   *Meaning*: Outlier-resistant wafer-relative scaling.
3. **`f460_wafer_zscore`**: $z_W(f_{460})$.
4. **`f336_wafer_zscore`**: $z_W(f_{336})$.
5. **`f33_wafer_zscore`**: $z_W(f_{33})$.
6. **`f109_wafer_zscore`**: $z_W(f_{109})$.
7. **`wafer_mean_f171`**: $\mu_W(f_{171})$. Provides the wafer's global process level.
8. **`wafer_std_f171`**: $\sigma_W(f_{171})$. Provides the wafer's intra-lot process variation.

---

## Family D: Signal Geometry & Location (8 features)

Let $b = [b_0, b_1, \dots, b_{1999}]$ be the 2,000-block analog sequence for a die. Let baseline threshold be $T = 100.0$.
Define anomalous amplitude: $a_i = \max(0, b_i - T)$.
Total anomalous area: $A_{\text{total}} = \sum_{i=0}^{1999} a_i$.

1. **`anomaly_peak_pos`**:
   $$i^* = \arg\max_{i} (b_i), \quad P_{\text{peak}} = \frac{i^*}{2000.0} \in [0, 1]$$
   *Meaning*: Where along the block sequence the maximum voltage deviation occurs.

2. **`anomaly_centroid`**:
   $$C_{\text{anom}} = \frac{\sum_{i=0}^{1999} i \cdot a_i}{A_{\text{total}} + 10^{-5}} \times \frac{1}{2000.0} \in [0, 1]$$
   *Meaning*: Center-of-mass location of analog damage along the memory array.

3. **`anomaly_early_ratio`**:
   $$R_{\text{early}} = \frac{\sum_{i=0}^{666} a_i}{A_{\text{total}} + 10^{-5}}$$
   *Meaning*: Proportion of anomaly in the first third of blocks.

4. **`anomaly_mid_ratio`**:
   $$R_{\text{mid}} = \frac{\sum_{i=667}^{1333} a_i}{A_{\text{total}} + 10^{-5}}$$
   *Meaning*: Proportion of anomaly in the central third of blocks.

5. **`anomaly_late_ratio`**:
   $$R_{\text{late}} = \frac{\sum_{i=1334}^{1999} a_i}{A_{\text{total}} + 10^{-5}}$$
   *Meaning*: Proportion of anomaly in the final third of blocks.

6. **`anomaly_asymmetry`**:
   $$\text{Asym} = \frac{\sum_{i=1000}^{1999} a_i - \sum_{i=0}^{999} a_i}{A_{\text{total}} + 10^{-5}}$$
   *Meaning*: Measures directional tilt (early-skewed vs late-skewed).

7. **`sharp_peak_ratio`**:
   Let window around peak be $\mathcal{W}_{\text{peak}} = [\max(0, i^* - 10), \min(1999, i^* + 10)]$:
   $$R_{\text{sharp}} = \frac{\sum_{i \in \mathcal{W}_{\text{peak}}} a_i}{A_{\text{total}} + 10^{-5}}$$
   *Meaning*: Differentiates sharp localized defect spikes ($R \approx 1$) from broad baseline drift ($R \ll 1$).

8. **`sustained_anomaly_flag`**:
   $$\mathbb{I}_{\text{sustained}} = \mathbb{I}\left(\text{longest\_run\_104} \ge 50\right)$$
   *Meaning*: Binary indicator that elevated readings persist continuously across $\ge 50$ blocks.

---

## Family E: Cross-Evidence Agreement & Concordance (5 features)

Constructs explicit multi-sensor concordance indicators combining electrical, spatial, and signal modalities:
- Standardized Electrical Anomaly: $E = \max(0, z_W(f_{171}))$.
- Spatial Defect Magnitude: $S = \text{density}_{5\times 5}$.
- Signal Anomaly Magnitude: $B = \log(1 + \text{cluster\_area}_{110})$.

1. **`elec_spatial_agreement`**:
   $$A_{ES} = E \times S$$
2. **`elec_signal_agreement`**:
   $$A_{EB} = E \times B$$
3. **`spatial_signal_agreement`**:
   $$A_{SB} = S \times B$$
4. **`concordant_evidence_count`**:
   $$N_{\text{conc}} = \mathbb{I}(E > 1.5) + \mathbb{I}(S > 0.10) + \mathbb{I}(B > 0.0)$$
   *Range*: $0, 1, 2, 3$.
5. **`full_concordance_flag`**:
   $$I_{\text{full}} = \mathbb{I}(N_{\text{conc}} = 3)$$
