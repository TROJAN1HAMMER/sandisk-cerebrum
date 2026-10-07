# EXP-05: Late Probability Stacking for Model B

## Objective
Investigate late probability-level fusion between tabular features (Model B2, 536 features) and raw 2,000-block sub-die signals (1D CNN), strictly avoiding raw CNN embedding concatenation which degraded PR-AUC in earlier research.

## Evaluated Variants
1. **Variant A**: Linear Probability Blend: $P_{\text{final}} = \alpha P_{\text{tabular}} + (1-\alpha) P_{\text{signal}}$ (Optimal $\alpha = 0.63$)
2. **Variant B**: Logit-Space Blend: $\text{logit}(P_{\text{final}}) = \alpha \text{logit}(P_{\text{tabular}}) + (1-\alpha) \text{logit}(P_{\text{signal}})$ (Optimal $\alpha = 0.68$)
3. **Variant C**: Logistic Regression Meta-Classifier trained out-of-fold on $[P_{\text{tabular}}, P_{\text{signal}}]$.

## Key Takeaway & Scientific Decision
**Decision: REJECTED_KEEP_FROZEN_B2**
- Baseline Frozen B2 Cascade F1: 0.5264 | PR-AUC: 0.4779
- Best Stacking Variant (Frozen_B2_Baseline) Cascade F1: 0.5264 | PR-AUC: 0.4779
- Delta F1: +0.0000 | Delta PR-AUC: +0.0000
