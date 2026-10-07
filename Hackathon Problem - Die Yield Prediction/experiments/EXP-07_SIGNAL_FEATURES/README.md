# EXP-07: Advanced 2,000-Block Signal Features

## Objective
Extend Model B2's sub-die block anomaly feature extraction by engineering compact, physically grounded feature families:
- **Statistical Moments**: Kurtosis, Skewness, Variance, RMS, MAD, Peak-to-RMS (6 features)
- **Run-Length Descriptors**: Cluster lengths and densities across thresholds >103, >104, >105, >110, >115 (30 features)
- **FFT Spectral Descriptors**: Dominant Frequency, Spectral Energy, Centroid, Entropy, Band Ratios (8 features)

## Ablation Architecture
1. **B2 Baseline**: 536 features (Frozen candidate)
2. **B2 + Statistical**: 542 features
3. **B2 + Run-Length**: 566 features
4. **B2 + FFT**: 544 features
5. **B2 + All**: 580 features

## Scientific Decision
**Decision: REJECTED_KEEP_FROZEN_B2**
- Baseline B2 Cascade: F1 = 0.5264 | PR-AUC = 0.4763
- Best Variant (B2_Baseline): F1 = 0.5264 | PR-AUC = 0.4763
- Delta F1: +0.0000 | Delta PR-AUC: +0.0000
