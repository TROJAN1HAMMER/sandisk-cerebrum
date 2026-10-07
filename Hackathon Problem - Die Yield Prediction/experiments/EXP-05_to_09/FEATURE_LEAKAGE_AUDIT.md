# Feature Leakage Audit & Data Hygiene Protocol
**Platform**: SanDisk Cerebrum Die Yield Intelligence  
**Scope**: Experimental Features for EXP-05 through EXP-09  
**Audit Standard**: Fab Production Rigor (Zero Downstream / Cross-Die Contamination)  

---

## 1. Governance Principles & Strict Rules

In high-volume semiconductor fab deployment, data leakage introduces catastrophic false confidence. Every feature evaluated across EXP-05 to EXP-09 must satisfy the following six zero-leakage conditions:

1. **Zero Target Access**: The feature calculation must never inspect, correlate with, or touch the post-test latent defect indicator `label`.
2. **Zero Downstream / Post-Test Information**: Only information strictly available at the end of the inline wafer electrical test / sub-die block read step can be utilized.
3. **Strict Spatial Isolation**: Spatial defect metrics must derive solely from pre-test electrical probe outcomes (`old_label == 1`), never from the post-test defect map (`label == 1`).
4. **Intra-Die Signal Isolation**: 2,000-block signal metrics must operate purely within each die's own 2,000 readings. No cross-die normalization or wafer-wide statistics computed using post-test dies may be applied.
5. **Wafer Partition Grouping**: All model weights, normalizers, calibration curves, and stacking meta-models must be fitted strictly inside wafer-grouped training partitions. Dies from the same wafer must never bridge train and evaluation sets.
6. **Train/Inference Parity**: Every feature extraction mathematical transform must be 100% deterministic and execute identically in offline batch training and real-time live inference.

---

## 2. Comprehensive Audit Matrix

| Feature Family / Component | Source Data | Target Access? | Cross-Die Info? | Post-Test Info? | Train/Infer Identical? | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A Parametric (500)** | In-line ATE electrical | NO | NO | NO | YES | **CERTIFIED** |
| **Spatial Coordinates** | Die row, die col | NO | NO | NO | YES | **CERTIFIED** |
| **Spatial Density (3x3, 5x5)** | Pre-test `old_label` | NO | Pre-test probe map only | NO | YES | **CERTIFIED** |
| **EXP-05 Stacking Meta-Models** | Tabular $P_B$ & Signal $P_{\text{cnn}}$ | NO | NO | NO | YES (Wafer-OOF) | **CERTIFIED** |
| **EXP-06 Model A Ensemble** | 506 Parametric + Spatial | NO | NO | NO | YES | **CERTIFIED** |
| **EXP-07 Statistical Moments** | 2,000 block array | NO | NO | NO | YES | **CERTIFIED** |
| **EXP-07 Run-Length Metrics** | 2,000 block array | NO | NO | NO | YES | **CERTIFIED** |
| **EXP-07 FFT Spectral Descriptors**| 2,000 block array | NO | NO | NO | YES | **CERTIFIED** |
| **EXP-08 TCN / Inception** | 2,000 block array | NO | NO | NO | YES | **CERTIFIED** |
| **EXP-09 Isotonic Calibration** | Model probabilities | NO | NO | NO | YES (Fitted on Train OOF) | **CERTIFIED** |

---

## 3. Mathematical Verification of EXP-07 Features

### 3.1 Statistical Descriptors
- **Kurtosis**: Pearson kurtosis $\kappa = \frac{\frac{1}{N}\sum (x_i - \mu)^4}{\sigma^4} - 3$. Strictly intra-die.
- **Skewness**: Fisher-Pearson coefficient $\gamma = \frac{\frac{1}{N}\sum (x_i - \mu)^3}{\sigma^3}$. Strictly intra-die.
- **RMS**: $\sqrt{\frac{1}{N}\sum x_i^2}$. Strictly intra-die.
- **Peak-to-RMS**: $\frac{\max(x)}{\text{RMS}}$. Strictly intra-die.
- **Mean Absolute Deviation (MAD)**: $\frac{1}{N}\sum |x_i - \text{median}(x)|$. Strictly intra-die.

### 3.2 Run-Length Descriptors
- Evaluated across thresholds $\tau \in \{103.0, 104.0, 105.0, 110.0, 115.0\}$:
  - Longest contiguous segment where $x_i > \tau$.
  - Count of contiguous anomalous clusters where $x_i > \tau$.
  - Mean cluster length above $\tau$.
  - 95th percentile cluster length.
  - Run density (fraction of array inside abnormal runs).
- *Audit Outcome*: All operations are strictly local 1D runs across index $i \in [0, 1999]$. No leakage.

### 3.3 Frequency-Domain (FFT) Descriptors
- Given 1D discrete Fourier transform $X_k = \sum_{n=0}^{N-1} x_n e^{-j 2\pi k n / N}$ computed via `np.fft.rfft(block_values - 100.0)`:
  - Power Spectral Density: $P_k = |X_k|^2$.
  - Dominant Frequency: $\arg\max_{k > 0} P_k$.
  - Spectral Energy: $\sum_{k > 0} P_k$.
  - Spectral Centroid: $\frac{\sum k P_k}{\sum P_k}$.
  - Spectral Entropy: $-\sum p_k \log(p_k)$ where $p_k = P_k / \sum P_k$.
  - Band Energy Ratios: Low band (bins 1–50) vs high band (bins 50–500).
- *Audit Outcome*: The DC component (baseline 100.0) is centered out, and spectra are computed exclusively on the die's own 2,000 points. Zero cross-die leakage.

---

## 4. Cross-Validation and Calibration Isolation

- Stacking meta-models in EXP-05 and calibration models in EXP-09 must **never** be fitted on the same data evaluated for performance.
- We will employ a 5-fold wafer-grouped cross-validation strategy on the 128 canonical training wafers:
  - Out-of-fold predictions $P_{\text{OOF}}$ will be generated for all training dies.
  - Fusion weights, meta-classifiers, and isotonic regression models will be fitted on $P_{\text{OOF}}$.
  - Holdout evaluation will be performed on the canonical 32 validation wafers.
  - The independent 40-wafer test set will remain **100% untouched** until final freeze.

**Conclusion**: All feature extraction routines and experiment workflows comply fully with zero-leakage standards.
