# Future Experiments Roadmap

The following candidate interventions were identified during EXP-03 and EXP-03B analysis but were strictly excluded from execution to preserve frozen confirmation integrity:

---

## 1. EXP-04: Probability Calibration & Cascade Score Harmonization
* **Hypothesis:** Model A and Model B output raw uncalibrated LightGBM tree margins. Expected Calibration Error (ECE) is currently $\sim 12\%$ on both models.
* **Proposed Intervention:** Train Isotonic Regression or Beta Calibration on out-of-fold training predictions for Model A and Model B. Evaluate whether aligned posterior defect probabilities improve ranking threshold selection and Brier score.

---

## 2. EXP-05: Adaptive Wavelet & Spectral Block Signal Decomposition
* **Hypothesis:** The 2,000-reading sensor series exhibits periodic high-frequency noise bursts in specific defect classes that moving average filters smooth out.
* **Proposed Intervention:** Compute 1D discrete wavelet transform (DWT) energy ratios (e.g. Haar / Daubechies detail coefficients $D_1, D_2, D_3$) to capture micro-crack signatures.

---

## 3. EXP-06: GroupKFold Cross-Validation on Full Wafer Fleet
* **Hypothesis:** Holdout evaluation on 32 wafers shows slight variance between sparse-failure and dense-failure wafers.
* **Proposed Intervention:** Execute 5-fold Wafer-Grouped Cross-Validation (GroupKFold over all 160 wafers) to establish narrow confidence intervals for cascade F1, PR-AUC, and escalation rates.
