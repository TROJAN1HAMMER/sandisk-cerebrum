# Final Architecture & Production Promotion Recommendation

## Executive Decision
**RETAIN B2 AS THE PRODUCTION DECISION ENGINE.**
**AUGMENT WITH EXP-08 TCN FOR ATTRIBUTION AND EXP-09 ISOTONIC CALIBRATION FOR RISK REPORTING.**

---

## 1. Scientific Justification

Across EXP-05 to EXP-09, each proposed technique was subjected to strict, controlled wafer-grouped validation against frozen B2 as the immutable control:

1. **EXP-05 (Late Probability Stacking)**:
   - *Verdict*: **REJECTED**.
   - *Evidence*: Stacking tabular B2 with 1D CNN probability degraded PR-AUC from 0.4779 to 0.4657 and triggered an unacceptable false alarm explosion.
2. **EXP-06 (Model A Ensemble)**:
   - *Verdict*: **REJECTED**.
   - *Evidence*: XGBoost and CatBoost ensembling failed to improve cascade recovery, collapsed compute savings from 94% down to 74%, and tripled inference latency (0.0066 -> 0.0208 ms/die).
3. **EXP-07 (Advanced Signal Features)**:
   - *Verdict*: **REJECTED**.
   - *Evidence*: Adding 44 statistical, run-length, and FFT descriptors diluted LightGBM tree splits, reducing Cascade F1 from 0.5264 to 0.5180.
4. **EXP-08 (Multi-Scale Dilated TCN)**:
   - *Verdict*: **ADOPTED AS FORENSIC ATTRIBUTION ENGINE ONLY**.
   - *Evidence*: Multi-Scale Dilated TCN achieved higher standalone sequence PR-AUC (0.0874 vs 0.0794) with fewer parameters (39,329 vs 41,153), making it superior for saliency and root-cause visualization. However, blending its output into the cascade decision degraded F1.
5. **EXP-09 (Isotonic Probability Calibration)**:
   - *Verdict*: **ADOPTED AS REPORTING CALIBRATION LAYER**.
   - *Evidence*: Slashed ECE by 97% (0.123 -> 0.003) and Brier Score by 43% (0.0417 -> 0.0238) while rank-preserving discrimination.

---

## 2. Production Promotion Rule Adherence
- B2 beat all 5 architectural candidates in internal grouped validation, ablation, latency, and operational simplicity.
- The independent 40-wafer benchmark confirms frozen B2 achieves **F1 = 0.5177, PR-AUC = 0.4808, 93.78% compute savings**, and **+37 net defects recovered**.
- Therefore, in strict adherence to Phase 13 of the program charter: **KEEP B2**.
