# EXP-06: Model A Ensemble (LightGBM + XGBoost + CatBoost)

## Objective
Evaluate whether ensembling three distinct gradient boosting architectures (LightGBM, XGBoost, and CatBoost) on 506 continuous parametric and spatial features improves Model A discriminability and downstream cascade defect recovery.

## Experimental Discipline
- Weights learned strictly via 5-fold wafer-grouped cross-validation on 128 training wafers (zero holdout tuning).
- Evaluated both standalone and through the full cascade with frozen Model B2 in gate $[0.295, 0.550]$.
- Measured training time, inference latency per die, and memory footprint.

## Summary of Results
- **LightGBM Alone**: Standalone PR-AUC = 0.4491, F1 = 0.4810 | Cascade F1 = 0.5114, PR-AUC = 0.4643
- **XGBoost Alone**: Standalone PR-AUC = 0.4596, F1 = 0.4906 | Cascade F1 = 0.5180, PR-AUC = 0.4723
- **CatBoost Alone**: Standalone PR-AUC = 0.4556, F1 = 0.2742 | Cascade F1 = 0.3652, PR-AUC = 0.4688
- **Ensemble (Weighted Avg)**: Standalone PR-AUC = 0.4596, F1 = 0.4906 | Cascade F1 = 0.5180, PR-AUC = 0.4723

## Scientific Decision
**Decision: REJECTED_KEEP_LIGHTGBM_MODEL_A**
