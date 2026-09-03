# Probability Calibration & Reliability Analysis
**Date**: September 3, 2026  
**Evaluation Set**: 27,972 canonical validation dies (32 holdout wafers)

---

## 1. Quantitative Calibration Metrics (Brier Score)

The Brier Score measures the mean squared error between predicted probabilities $P(Y=1)$ and empirical binary outcomes $Y \in \{0, 1\}$:

$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (P_i - Y_i)^2$$

| Architecture | Gating Window | Brier Score (Lower is Better) | Calibration Status |
| :--- | :---: | :---: | :--- |
| **Model A Baseline** | None | `0.04561` | Well-calibrated at tails, mild underconfidence in 0.30–0.50 band |
| **Model B (B4 Full)** | Unrestricted | `0.04546` | Slightly better global mean calibration, wider variance |
| **Selective Cascade (Gate B)** | $[0.25, 0.55]$ | **`0.04175`** | **Best calibrated system (-8.46% Brier reduction vs Model A)** |
| **Selective Cascade (Gate C)** | $[0.30, 0.60]$ | **`0.04282`** | Highly calibrated (-6.12% Brier reduction vs Model A) |

---

## 2. Reliability Curve Interpretation

Reliability curves were generated across 10 empirical probability deciles and saved in [`visualizations/probability_calibration.png`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/visualizations/probability_calibration.png):

1. **Low Probability Regime ($P < 0.20$)**:
   - Both Model A and Model B exhibit near-perfect alignment with the diagonal line.
   - For dies predicted with $P \in [0.00, 0.10]$, empirical failure rate is $0.8\%$.
2. **Intermediate Ambiguity Regime ($0.25 \le P \le 0.55$)**:
   - Model A tends to overestimate risk slightly due to parametric noise overlap.
   - The Selective Cascade (Gate B / Gate C) uses sub-die block readings to resolve this ambiguity, pulling genuine passing dies down and boosting genuine defect clusters up, flattening the calibration curve toward the ideal 45° diagonal.
3. **High Probability Regime ($P > 0.60$)**:
   - High-risk predictions achieve empirical failure rates exceeding $85\%$, confirming high precision and reliable flagging for industrial scrap/rework.

---

## 3. Engineering Conclusion

No post-hoc Platt scaling or isotonic regression is required. The Selective Cascade natively improves Brier calibration from **0.04561 to 0.04175** because sub-die block evidence resolves marginal parametric ambiguities without distorting extreme probabilities.
