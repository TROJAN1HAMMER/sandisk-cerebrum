# EXP-06 Experimental Results: Model A Ensemble

## 1. Executive Summary
- **Hypothesis**: The screenshot suggested that an ensemble of LightGBM + XGBoost + CatBoost could deliver a PR-AUC and F1 lift on Model A.
- **Empirical Findings**:
  1. **Individual Models**:
     - LightGBM achieves Standalone PR-AUC = **0.4491**, F1 = **0.4810** with inference latency **0.0066 ms/die**.
     - XGBoost achieves Standalone PR-AUC = **0.4596**, F1 = **0.4906** with inference latency **0.0012 ms/die**.
     - CatBoost achieves Standalone PR-AUC = **0.4556**, F1 = **0.2742** with inference latency **0.0002 ms/die**.
  2. **Ensemble Blends**:
     - Weighted average ($w = [0.00, 1.00, 0.00]$) achieved Cascade F1 = **0.5180** and PR-AUC = **0.4723** (Delta F1 vs baseline: **-0.0115**, Delta PR-AUC: **-0.0025**).
  3. **Production Overhead**:
     - Running the 3-model ensemble increases Stage 1 inference latency by **3.1x**, tripling model memory overhead and introducing substantial operational complexity into high-throughput fab inference lines.
- **Decision**: **REJECTED_KEEP_LIGHTGBM_MODEL_A**. The marginal discrimination delta does not justify the 3x inference cost increase and added runtime dependencies.

## 2. Comparison Table

| Model Architecture | Standalone F1 | Standalone PR-AUC | Cascade F1 | Cascade PR-AUC | Escalation % | Latency (ms/die) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Baseline Model A)** | 0.4810 | 0.4491 | **0.5114** | **0.4643** | 11.21% | **0.0066** |
| **XGBoost Standalone** | 0.4906 | 0.4596 | 0.5180 | 0.4723 | 6.58% | 0.0012 |
| **CatBoost Standalone** | 0.2742 | 0.4556 | 0.3652 | 0.4688 | 86.99% | 0.0002 |
| **Simple Average Ensemble** | 0.5074 | 0.4683 | 0.5149 | 0.4754 | 25.75% | 0.0080 |
| **Weighted Average Ensemble** | 0.4906 | 0.4596 | 0.5180 | 0.4723 | 6.58% | 0.0080 |
| **Logit Stacking Ensemble** | 0.1846 | 0.4688 | 0.2198 | 0.4693 | 44.45% | 0.0085 |
