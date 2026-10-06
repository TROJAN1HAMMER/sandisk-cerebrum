# Staged Experiment Roadmap & Scientific Governance

**Location:** `audit/model_forensics/experiment_roadmap.md`  
**Date:** October 6, 2026  
**Auditor:** Antigravity Forensic ML Engine  
**Status:** COMPLETE & GOVERNED  

---

## 1. Executive Protocol & Governance Principles

Any future model training phase on the SanDisk Cerebrum pipeline must adhere to strict scientific governance. Optimization for a single headline metric (such as overall accuracy or global F1) is strictly prohibited because semiconductor yield intelligence requires joint satisfaction of:
1. **Defect Escape Suppression (Recall / FN minimization):** Missing subtle defects causes catastrophic field reliability failures.
2. **False Alarm Minimization (Precision / FP control):** False alarms cause needless destruction of good silicon and lower factory yield.
3. **Wafer-Level Robustness (Worst-Wafer Stability):** Global metrics must not hide catastrophic failures on individual production wafers.
4. **Computational Practicability (Cascade Latency):** Sub-millisecond die screening to meet fab test equipment cycle times.
5. **Zero Data Leakage:** Unbreakable wafer isolation across all train, validation, and holdout folds.

---

## 2. Multi-Metric Acceptance Scorecard Criteria

No future model candidate shall be accepted into production or deemed superior to the current baseline unless it meets the following multi-gate criteria:

| Dimension | Primary Metric | Baseline Value | Required Target / Guardrail | Rejection Condition |
| :--- | :--- | :---: | :---: | :--- |
| **Defect Detection** | Cascade Recall | $36.84\%$ ($389$ TPs) | $\ge 42.00\%$ ($\ge 443$ TPs) | Recall drops below $36.00\%$ |
| **Defect Recovery** | Gross FN Recovery | $37$ dies ($5.29\%$) | $\ge 60$ dies ($\ge 8.5\%$) | Gross recovery $< 35$ dies |
| **Yield Protection** | Cascade Precision | $81.38\%$ ($89$ FPs) | $\ge 80.00\%$ | Precision drops below $75.00\%$ |
| **Overall Balance** | Cascade F1 Score | $0.5072$ | $\ge 0.5250$ | F1 drops below $0.5050$ |
| **Ranking Quality** | Cascade PR-AUC | $0.4677$ | $\ge 0.4850$ | PR-AUC drops below $0.4650$ |
| **Discriminability** | Cascade ROC-AUC | $0.7990$ | $\ge 0.8100$ | ROC-AUC drops below $0.7900$ |
| **Wafer Robustness** | Worst-Wafer Recall | $18.18\%$ | $\ge 25.00\%$ | Worst-wafer recall $< 18.00\%$ |
| **Wafer Stability** | Recall Std Dev | $11.92\%$ | $\le 10.00\%$ | Recall dispersion $> 15.00\%$ |
| **Calibration** | Expected Calib Error | $0.12053$ | $\le 0.06000$ | ECE $> 0.15000$ |
| **Throughput** | Compute Saved | $88.14\%$ | $\ge 80.00\%$ | Compute saved $< 75.00\%$ |
| **Single-Die Latency**| Cascade Latency | $0.0066$ ms | $\le 0.0200$ ms | Single-die latency $> 0.1000$ ms |

---

## 3. Staged Experiment Roadmap

The proposed experiment roadmap is strictly sequential, building from zero-cost data and feature bug fixes to advanced modeling:

```text
Stage 0 (Zero Retraining / Baseline Reproduction)
   │
Stage 1 (Data Preprocessing & Feature Hygiene)
   │
Stage 2 (Signal Feature Expansion & Dead Feature Pruning)
   │
Stage 3 (LightGBM Hyperparameter Tuning & Wafer Cross-Validation)
   │
Stage 4 (Cascade Gating & Joint Threshold Optimization)
   │
Stage 5 (Probability Recalibration)
   │
Stage 6 (Advanced Multi-Resolution Architecture / Hybrid Modeling)
   │
Stage 7 (Final Holdout Evaluation & Deployment Lock)
```

### Stage 0: Baseline Verification & Environment Lock
* **Objective:** Verify 100% bitwise metric reproducibility on `results/val_die_diagnostics.csv` and canonical split.
* **Validation:** Assert $357$ TPs, $699$ FNs, $42$ FPs, $26,874$ TNs for Model A; $+37$ recovered FNs, $89$ FPs for Cascade Gate B.
* **Cost:** Zero compute.

### Stage 1: Spatial Preprocessing Fix (Highest ROI)
* **Objective:** Fix the `src/spatial.py` call sequence by passing `full_df=df` to restore genuine pre-test `density_3x3` and `density_5x5` features.
* **Intervention:** Re-train Model A on 128 canonical wafers with restored spatial density features.
* **Expected Result:** Boost Model A baseline F1 from $0.4907$ to $\approx 0.5100+$, recovering spatial defect clusters.
* **Risk:** Zero risk; uses pre-test information strictly.

### Stage 2: Signal Feature Expansion & Dead Feature Pruning
* **Objective:** Expand the high-leverage block anomaly signal while pruning uninformative dimensions.
* **Interventions:**
  * Add `cluster_area_103`, `cluster_area_105`, `cluster_energy_l2`.
  * Prune zero-gain features: `longest_run_104`, `cluster_width`, `b_min`.
* **Validation:** Re-train Model B on 128 canonical wafers; benchmark PR-AUC against B4 ($0.4659$).
* **Expected Result:** Model B Standalone PR-AUC boosts from $0.4659$ to $\approx 0.4850+$.

### Stage 3: LightGBM Hyperparameter Search with 5-Fold StratifiedGroupKFold
* **Objective:** Optimize tree capacity, regularization, and subsampling across wafer groups.
* **Search Space:**
  * `num_leaves`: $[31, 63, 127]$
  * `min_child_samples`: $[20, 50, 100]$
  * `colsample_bytree`: $[0.6, 0.7, 0.8]$
  * `scale_pos_weight`: $[4.0, 6.0, 8.0, 10.0]$
  * `reg_alpha`, `reg_lambda`: $[0.01, 0.1, 1.0]$
* **Group Isolation:** Grouped strictly by `wafer_id`. Zero die-level splitting.

### Stage 4: Asymmetric Cascade Gating & Joint Threshold Optimization
* **Objective:** Solve the $P_A < 0.25$ gating bottleneck.
* **Interventions:**
  * Sweep lower gate bound $L_G \in [0.15, 0.28]$ and upper gate bound $U_G \in [0.50, 0.65]$.
  * Jointly optimize $(\tau_A, \tau_B, L_G, U_G)$ against cost objective:
    $$\text{Cost} = C_{\text{FN}} \cdot \text{FN} + C_{\text{FP}} \cdot \text{FP} + C_{\text{comp}} \cdot N_{\text{escalated}}$$
* **Expected Result:** Recover $25-40$ additional FNs while containing new FPs under $100$.

### Stage 5: Probability Recalibration
* **Objective:** Correct the $5-7\times$ overconfidence distortion caused by `scale_pos_weight = 10.0`.
* **Method:** Fit out-of-fold Isotonic Regression or Beta Calibration on validation wafer predictions.
* **Expected Result:** Lower Brier score from $0.04175$ to $\le 0.03000$; drop ECE below $0.05000$.

### Stage 6: Advanced Signal Modeling / 1D Dilated Residual Block CNN
* **Objective:** Revisit deep learning representation without diluting tabular LightGBM splits.
* **Method:** Replace global pooling with localized Max-Attention over 2,000 blocks. Extract a single continuous anomaly risk score (1D scalar) rather than 32 unconstrained embeddings, and pass this single score to Model B.
* **Expected Result:** High signal-to-noise ratio without overfitting.

### Stage 7: Independent Competition Holdout Validation
* **Objective:** Evaluate the locked champion cascade on the 40 unseen wafers in `input/test.csv`.
* **Verification:** Verify that test-set performance aligns within $\pm 3\%$ of validation scorecard without degradation.

---

## 4. Experiment Logging Specification

Every future experiment MUST write an immutable JSON record to `results/experiments/` containing:

```json
{
  "experiment_id": "EXP-2026-09-001",
  "timestamp": "2026-10-06T21:30:00Z",
  "git_commit": "abcdef123456",
  "dataset_version": "1.0.0",
  "split_artifact": "results/canonical_wafer_split.json",
  "model_architecture": "LightGBM + Selective Cascade",
  "features_count_A": 506,
  "features_count_B": 526,
  "hyperparameters": {},
  "thresholds": {
    "tau_A": 0.51,
    "tau_B": 0.5376,
    "gate_lower": 0.25,
    "gate_upper": 0.55
  },
  "metrics_global": {
    "f1": 0.5072,
    "pr_auc": 0.4677,
    "roc_auc": 0.7990,
    "precision": 0.8138,
    "recall": 0.3684,
    "brier": 0.04175,
    "ece": 0.12053
  },
  "confusion_matrix": {
    "tp": 389,
    "fn": 667,
    "fp": 89,
    "tn": 26827
  },
  "cascade_performance": {
    "gross_fn_recovered": 37,
    "lost_tps": 5,
    "net_tp_gain": 32,
    "new_fps": 47,
    "escalation_rate_pct": 11.86,
    "compute_saved_pct": 88.14
  },
  "wafer_robustness": {
    "mean_wafer_recall": 0.3856,
    "median_wafer_recall": 0.3738,
    "worst_wafer_recall": 0.1818,
    "wafer_recall_std": 0.1192
  },
  "decision": "ACCEPTED / REJECTED",
  "justification": "Detailed rationale"
}
```
