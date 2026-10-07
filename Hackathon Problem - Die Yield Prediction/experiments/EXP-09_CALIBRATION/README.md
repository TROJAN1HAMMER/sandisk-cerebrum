# EXP-09: Isotonic Probability Calibration

## Objective
Measure the exact empirical effect of non-parametric Isotonic Calibration on probability reliability (Brier score, Expected Calibration Error) and operational classification metrics (Precision, Recall, F1, F2).

## Core Principle
Calibration is a **probability reliability** technique, not a discrimination enhancement technique. It aligns predicted risk with empirical event frequency ($P(\hat{y}) \approx P(y=1)$) without altering the underlying ROC-AUC or ranking capability.

## Key Empirical Results
- **Model B2 ECE**: Uncalibrated = 0.11971 $\to$ Calibrated = 0.00912 (Improved reliability)
- **Model B2 Brier Score**: Uncalibrated = 0.04571 $\to$ Calibrated = 0.02558
- **Cascade F1**: Uncalibrated = 0.5264 $\to$ Calibrated = 0.5217
- **ROC-AUC**: 0.7986 $\to$ 0.8012 (Strictly preserved)
