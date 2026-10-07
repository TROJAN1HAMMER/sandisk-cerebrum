# EXP-10: Feature Audit — Context-Aware Die Intelligence

**Experiment:** EXP-10 Context-Aware Die Intelligence  
**Platform:** SanDisk Cerebrum Die Yield Prediction  
**Status:** Audit & Architecture Specification  
**Baseline:** Frozen Production B2 Selective Cascade (506 Model A features, 536 Model B2 features)  
**Date:** October 2026  

---

## 1. Existing Contextual Information in Current B2

The production B2 system operates a total of 536 tabular features. Prior to EXP-10, their context representation was audited as follows:

| Category | Existing Features | Contextual Scope | Limitation |
| :--- | :---: | :--- | :--- |
| **Electrical Parametric** | `feature_1` ... `feature_500` (500) | Strictly die-isolated. Raw scalar test readouts. | Zero neighbourhood context. Zero wafer-relative scaling. Global values treated identically across distinct wafer runs. |
| **Spatial Coordinates** | `die_row`, `die_col` (2) | Absolute coordinate grid indices. | Lacks orientation, aspect ratio, or relative cluster topology. |
| **Radial / Edge Distance** | `radial_dist`, `edge_dist` (2) | Distance from wafer center and boundary. | Macro-geometric location only; does not capture local neighbourhood structure. |
| **Pre-Test Defect Density** | `density_3x3`, `density_5x5` (2) | Isotropic local fraction of screening rejects (`old_label`). | Measures defect volume only; blind to shape (linear scratch vs circular patch vs isolated die) and blind to electrical neighbor values. |
| **Legacy Block Features** | 20 features (`b_mean`, `b_std`, `cluster_area`, `max_roll50`, etc.) | Summary statistics of the 2,000-block analog sequence. | Averages over time/space; does not localize *where* in the memory array the anomaly occurs or its physical geometry. |
| **B2 Anomaly Descriptors** | 10 features (`cnt_110`, `max_roll20`, `cluster_area_110`, etc.) | Extreme analog voltage excursions. | Captures intensity and local rolling max, but lacks multi-scale geometry and location asymmetry. |

---

## 2. Missing Contextual Information

The current model asks:
> *"How abnormal is this die in absolute global feature space?"*

It fails to ask:
1. **Local Electrical Neighbourhood Contrast**:
   - Is this die's electrical measurement significantly divergent from the dies physically adjacent to it? (An isolated electrical spike in a sea of normal dies vs a uniform regional gradient).
2. **Spatial Topology & Morphology**:
   - What is the geometric orientation and shape of nearby defects? Is the defect pattern an isolated die, a vertical scratch, a horizontal track, or an isotropic cluster?
3. **Wafer-Relative Baseline Normalization**:
   - Fabrication tools drift across runs. An electrical value of $+1.5\sigma$ globally might be right at the median for a hot wafer, or $+4.0\sigma$ on a cool wafer. The model lacks wafer-level z-scores and robust deviations.
4. **Signal Geometry & Location**:
   - Where along the 2,000-block wordline/bitline structure is the anomaly situated? Early blocks (near page buffers), middle blocks, or late blocks (near wordline drivers)? Is it sharp ($<10$ blocks) or distributed ($>100$ blocks)?
5. **Cross-Evidence Concordance / Agreement**:
   - Do the electrical probe, physical neighborhood, spatial pre-test cluster, and 2,000-block signal simultaneously agree on the anomaly, or is only one noisy sensor firing?

---

## 3. Candidate New Feature Families for EXP-10

To prevent dimensionality explosion and split dilution, we design compact, physically grounded feature families:

### Family A: Local Physical Neighbourhood (7 features)
Targeting the top electrical signals (`feature_171`, `feature_460`, `feature_336`, `feature_33`, `feature_109`):
1. `valid_neighbors_3x3`: Count of valid active dies in the $3\times 3$ window (boundary vs core indicator).
2. `f171_local_mean_3x3`: Mean of primary electrical driver (`feature_171`) across immediate 8 neighbours.
3. `f171_local_diff_3x3`: Difference: $\text{die.f171} - \text{local\_mean}$.
4. `f171_local_zscore_3x3`: Local deviation normalized by local standard deviation: $(\text{die} - \mu) / (\sigma + \epsilon)$.
5. `f460_local_diff_3x3`: Difference: $\text{die.f460} - \text{local\_mean}$.
6. `f336_local_diff_3x3`: Difference: $\text{die.f336} - \text{local\_mean}$.
7. `local_electrical_contrast`: Composite norm of local electrical outlier vectors across top drivers.

### Family B: Spatial Topology & Morphology (6 features)
Analyzing the geometric structure of pre-test screening rejects in the $5\times 5$ window:
1. `h_defect_density_5x5`: Horizontal defect concentration (horizontal axis scratch indicator).
2. `v_defect_density_5x5`: Vertical defect concentration (vertical axis track indicator).
3. `diag_defect_density_5x5`: Diagonal defect concentration.
4. `cluster_aspect_ratio`: $\log(\text{h\_density} + \epsilon) - \log(\text{v\_density} + \epsilon)$ (morphological elongation).
5. `isolated_defect_flag`: Indicator that the die has pre-test neighbor defects $=0$ while surrounding outer rings have defects.
6. `dense_cluster_flag`: Indicator that $5\times 5$ defect density exceeds $0.50$ (heavy defect patch).

### Family C: Wafer-Relative Context (8 features)
Computing within-wafer relative deviations and robust z-scores for top electrical drivers:
1. `f171_wafer_zscore`: $(x_{171} - \mu_{W, 171}) / (\sigma_{W, 171} + \epsilon)$.
2. `f171_wafer_robust_z`: $(x_{171} - \text{median}_{W, 171}) / (\text{MAD}_{W, 171} + \epsilon)$.
3. `f460_wafer_zscore`: $(x_{460} - \mu_{W, 460}) / (\sigma_{W, 460} + \epsilon)$.
4. `f336_wafer_zscore`: $(x_{336} - \mu_{W, 336}) / (\sigma_{W, 336} + \epsilon)$.
5. `f33_wafer_zscore`: $(x_{33} - \mu_{W, 33}) / (\sigma_{W, 33} + \epsilon)$.
6. `f109_wafer_zscore`: $(x_{109} - \mu_{W, 109}) / (\sigma_{W, 109} + \epsilon)$.
7. `wafer_mean_f171`: Baseline level of the wafer for primary electrical driver.
8. `wafer_std_f171`: Variance level of the wafer for primary electrical driver.

### Family D: Signal Geometry & Location (8 features)
Extracting spatial geometry along the 2,000-block time-series signal:
1. `anomaly_peak_pos`: Normalized index in $[0, 1]$ where $\max(\text{block})$ occurs.
2. `anomaly_centroid`: Center of mass of block readings above $100$: $\sum (i \cdot \max(0, b_i - 100)) / (\sum \max(0, b_i - 100) + \epsilon)$.
3. `anomaly_early_ratio`: Fraction of anomalous area located in blocks $0 \dots 666$.
4. `anomaly_mid_ratio`: Fraction of anomalous area located in blocks $667 \dots 1333$.
5. `anomaly_late_ratio`: Fraction of anomalous area located in blocks $1334 \dots 1999$.
6. `anomaly_asymmetry`: Left vs right area asymmetry: $(A_{\text{late}} - A_{\text{early}}) / (A_{\text{total}} + \epsilon)$.
7. `sharp_peak_ratio`: Area in $\pm 10$ blocks around peak divided by total anomaly area (sharp vs diffuse).
8. `sustained_anomaly_flag`: Binary indicator that anomalous readings span $>100$ continuous blocks.

### Family E: Cross-Evidence Agreement & Concordance (5 features)
Measuring whether orthogonal detection sensors corroborate each other:
1. `elec_spatial_agreement`: Product of normalized electrical anomaly magnitude $\times$ local spatial defect density.
2. `elec_signal_agreement`: Product of electrical anomaly magnitude $\times$ block cluster area.
3. `spatial_signal_agreement`: Product of spatial defect density $\times$ block cluster area.
4. `concordant_evidence_count`: Count of active anomaly sensors ($0 \dots 3$) exceeding their 90th percentile thresholds.
5. `full_concordance_flag`: Binary indicator ($1$ if electrical, spatial, and signal simultaneously flag anomalous).

---

## 4. Potential Leakage Risks & Strict Prevention

| Risk Factor | Mechanism of Leakage | Strict Prevention in EXP-10 |
| :--- | :--- | :--- |
| **Target Label in Neighbours** | Using neighbour ground-truth `label` in local mean or density. | **STRICT PROHIBITION**: Local neighbour statistics use ONLY raw input measurements (`feature_k`) or probe screening flags (`old_label`). Target `label` is NEVER queried. |
| **Wafer Normalization Across Splits** | Computing wafer statistics using data pooled across train and test. | **PER-WAFER ISOLATION**: Wafer means and standard deviations are computed strictly inside each wafer boundary independently. In testing, each test wafer computes its own statistics. |
| **Future / Downstream Test Leakage** | Using post-package or downstream burn-in telemetry. | All features use solely the probe electrical test, die coordinates, and 2,000-block reading available at Sort-1 inference. |
| **Out-of-Grid Padding Artifacts** | Grid padding assuming zero outside wafer perimeter. | All filters normalize strictly by the count of `valid_grid` dies in the local window. |

---

## 5. Features That MUST NOT Be Added

1. **Blind expansion across all 500 electrical features**:
   - Calculating 3x3 local mean, std, z-score, and wafer stats for all 500 features would add $500 \times 4 = 2,000$ features.
   - EXP-07 proved that high-dimensional feature expansion causes severe tree split dilution and drops Cascade F1.
   - We strictly restrict Family A and Family C to the top 5 verified electrical drivers (`feature_171`, `feature_460`, `feature_336`, `feature_33`, `feature_109`).
2. **Generic FFT / Kurtosis / Run-length features**:
   - Explicitly banned per Section 7 of the user directive; already rejected in EXP-07.
3. **Target-encoded wafer IDs**:
   - Encoding wafer historical yield risks leaks holdout wafer distributions and overfits training wafers.
