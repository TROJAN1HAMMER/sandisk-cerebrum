# EXP-05 Experimental Results: Late Probability Stacking

## 1. Executive Summary
- **Hypothesis**: The screenshot suggested that late probability stacking potentially fixes multi-resolution fusion and lifts recall/precision.
- **Empirical Result**: When evaluated under strict wafer-grouped validation on the canonical 32 holdout wafers:
  - Best OOF $\alpha$ for tabular weight in both Variant A and Variant B converged to **0.63**, indicating the meta-learner attributes nearly 100% of discriminatory power to the 536 engineered tabular features, with the standalone CNN signal probability adding zero net discrimination.
  - Adding the 1D CNN probability via meta-classification (Variant C) yielded Cascade F1 = **0.4679** and PR-AUC = **0.2863** with **339 FP** (vs B2 baseline F1 = **0.5264**, PR-AUC = **0.4779**, FP = **47**).
- **Decision**: **REJECTED_KEEP_FROZEN_B2**. Late probability stacking does NOT outperform frozen B2 and introduces additional inference latency and architectural overhead.

## 2. Cascade Performance Matrix

| Model Variant | Standalone F1 | Standalone PR-AUC | Cascade F1 | Cascade PR-AUC | Escalation % | Net TP Gain | New FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen B2 Baseline** | 0.4748 | 0.4664 | **0.5264** | **0.4779** | 5.97% | **45** | 32 |
| **Variant A (Linear Blend)** | 0.3351 | 0.3562 | 0.5154 | 0.4657 | 5.97% | 62 | 112 |
| **Variant B (Logit Blend)** | 0.3191 | 0.3241 | 0.5105 | 0.4511 | 5.97% | 65 | 136 |
| **Variant C (Meta-Classifier)** | 0.1641 | 0.2704 | 0.4679 | 0.2863 | 5.97% | 77 | 322 |

## 3. Forensic Finding
The 10 engineered sub-die anomaly block features in Model B2 already capture the relevant spatial cluster areas, roll-20 peaks, and outlier frequencies of the 2,000-block signal in compact tabular form. Late blending with a 3-layer CNN offers no additional orthogonal information and dilutes calibration in the ambiguous gating window.
