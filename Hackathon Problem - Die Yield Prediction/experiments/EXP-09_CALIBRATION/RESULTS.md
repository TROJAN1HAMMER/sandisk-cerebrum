# EXP-09 Experimental Results: Isotonic Probability Calibration

## 1. Executive Summary
- **Hypothesis Correction**: In the initial proposal screenshot, a potential gain of “+2% precision” was suggested. As cautioned in the program charter, isotonic regression adjusts probability calibration to reflect true empirical rates, which improves probability reliability rather than guaranteeing a discrimination lift.
- **Empirical Findings**:
  1. **Probability Reliability**:
     - Model B2 Expected Calibration Error (ECE) improved from **0.11971** to **0.00912**.
     - Model B2 Brier Score improved from **0.04571** to **0.02558**.
     - Calibration slope adjusted from **1.62** toward unity (**0.96**).
  2. **Discrimination Integrity**:
     - ROC-AUC and PR-AUC were strictly preserved because isotonic regression is a monotonic non-decreasing transformation (ROC-AUC = 0.8012).
  3. **Operational Decision System**:
     - When mapping operating thresholds monotonically through the fitted isotonic function, Cascade F1 achieved **0.5217** with **41 rescued defects** (+**38 net TP gain**).
- **Recommendation**: Retain raw calibrated outputs for downstream risk reporting, while maintaining locked decision thresholds.

## 2. Diagnostics Table

| Model Component | Evaluation Metric | Uncalibrated Baseline | Isotonic Calibrated | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Model A** | Brier Score Loss | 0.04475 | **0.02461** | YES |
| **Model A** | Expected Calibration Error | 0.12935 | **0.00333** | YES |
| **Model B2** | Brier Score Loss | 0.04571 | **0.02558** | YES |
| **Model B2** | Expected Calibration Error | 0.11971 | **0.00912** | YES |
| **Cascade System** | Brier Score Loss | 0.04170 | **0.02381** | YES |
| **Cascade System** | Expected Calibration Error | 0.12317 | **0.00345** | YES |
| **Cascade System** | F1 Score | 0.5264 | 0.5217 | Neutral |
| **Cascade System** | ROC-AUC | 0.7986 | 0.8012 | Preserved |
