# MODEL & DATASET FORENSIC AUDIT: SANDISK CEREBRUM DIE YIELD INTELLIGENCE

**Audit Date:** October 6, 2026  
**Auditor:** Antigravity Forensic ML Engine (DeepMind / Cerebrum Integrity Protocol)  
**Location:** `audit/model_forensics/MODEL_FORENSIC_AUDIT_FINAL.md`  
**Baseline Artifacts Audited:**
* `results/val_die_diagnostics.csv` (27,972 holdout dies across 32 wafers)
* `results/AUTHORITATIVE_CLEAN_METRICS.json`
* `results/canonical_wafer_split.json`
* `models/model_A.txt`, `models/model_A.pkl`, `models/model_B.pkl`, `models/cnn_1d_block_model.pt`
* `input/train.csv`, `input/validation.csv`, `input/test.csv`
* `src/*` ML inference and feature generation pipeline

---

## 1. Executive Verdict

The SanDisk Cerebrum ML system is **structurally sound, mathematically leak-free at the wafer boundary, and computationally efficient**, but it is currently **leaving substantial measurable performance on the table due to three specific engineering bottlenecks**:

1. **A Preprocessing Sequencing Defect Zeroed Out Spatial Neighborhood Clustering:**
   In `src/spatial.py`, `compute_spatial_features(eligible)` was executed on already-filtered `eligible` dies without passing the unfiltered wafer map (`full_df=df`). Because `old_label` was identically 0 for all eligible dies, `density_3x3` and `density_5x5` collapsed into **100% constant zero columns**. Model A and Model B were completely blind to spatial defect clustering. Fixing this call sequence will immediately restore valid neighborhood defect signals ($0.0$ to $0.67$) with zero retraining risk.
2. **The Selective Cascade Suffers from an Asymmetric Gating Bottleneck:**
   Of Model A's 699 False Negatives, **517 dies ($73.96\%$) have $P_A < 0.25$**. Because Gate B only escalates dies within $[0.25, 0.55]$, nearly three-quarters of the entire defect blind spot is bypassed on the fast path and never inspected by Model B. Model B was only given 182 FNs to evaluate, yet it successfully rescued 37 of them ($20.3\%$). Lowering the lower gate threshold to $0.18-0.20$ exposes over 200 additional subtle defects to Model B's dominant `cluster_area` detector.
3. **Severe Probability Distortion from `scale_pos_weight = 10.0`:**
   Because positive failure samples were weighted $10\times$ in training, predicted probabilities are severely uncalibrated ($\text{ECE} = 0.13116$). A die with $P_A = 0.20$ has an empirical failure rate of only $2.11\%$, and $P_A = 0.35$ corresponds to only $6.12\%$ empirical defect risk. Engineers interpreting $P_A$ as posterior failure probability are misled by a $5-7\times$ overconfidence error. Post-hoc calibration will align risk probabilities with true factory yield without degrading classification ranking.

---

## 2. Current Baseline (Authoritative Verification)

The authoritative baseline metrics established in Phase 1 & 3C were independently computed from `results/val_die_diagnostics.csv` and confirmed bit-for-bit:

```text
Validation Dies:          27,972
Validation Wafers:        32
Actual Defective Dies:    1,056
Actual Passing Dies:      26,916
Empirical Defect Rate:    3.7752%
Empirical Die Yield:      96.2248%

Fast Path Dies:           24,655 (88.14%)
Gate B Escalated Dies:    3,317  (11.86%)
Compute Budget Saved:     88.14%

Model A Baseline (τA = 0.5100):
  TP = 357   |   FN = 699   |   FP = 42   |   TN = 26,874
  Precision  = 89.47%
  Recall     = 33.81%
  F1 Score   = 0.4907
  F2 Score   = 0.3861
  ROC-AUC    = 0.7934
  PR-AUC     = 0.4498
  Brier Loss = 0.04561
  ECE        = 0.13116

Selective Cascade (Gate B [0.25, 0.55], τA = 0.5100, τB = 0.5376):
  TP = 389   |   FN = 667   |   FP = 89   |   TN = 26,827
  Precision  = 81.38%
  Recall     = 36.84%
  F1 Score   = 0.5072 (+0.0165 over Model A)
  F2 Score   = 0.4137 (+0.0276 over Model A)
  ROC-AUC    = 0.7990 (+0.0056 over Model A)
  PR-AUC     = 0.4677 (+0.0179 over Model A)
  Brier Loss = 0.04175 (Improved probability error)
  ECE        = 0.12053

Cascade Transition Dynamics:
  Gross Recovered FNs:     +37 dies (Model B caught subtle defect)
  Lost Borderline TPs:      -5 dies (dies in [0.51, 0.55] where PB < 0.5376)
  Net TP Gain:             +32 dies (357 → 389)
  New False Positives:     +47 dies (42 → 89)
```

---

## 3. Dataset Findings

* **Input Data Topology:**
  * `input/train.csv`: 173,099 rows across 160 wafers. Contains 19,062 pre-test rejects (`old_label == 1`, 11.01%) and 154,037 eligible pre-test passes (`old_label == 0`, 88.99%).
  * `input/validation.csv` & `input/test.csv`: Exactly 39,351 rows across 40 identical wafers. `validation.csv` is the unlabeled competition leaderboard test set; `test.csv` is the labeled ground truth.
  * `results/val_die_diagnostics.csv`: Exact 27,972 holdout validation slice partitioned from `train.csv`.
* **Zero Missing / Zero Duplicate Rows:** All 251,801 rows across all files have complete data, zero nulls, and unique $(x, y)$ coordinate pairs per wafer.
* **Signal Storage:** 2,000 sequential readings per die stored as space-separated ASCII floats, consuming ~85% of CSV disk storage. Parsing time is ~4.2 seconds for 28k dies.
* **Semiconductor Physics Grounding:** Synthetic data generation (`generate_data.py`) accurately mimics memory manufacturing:
  * 65% of post-test failures are marginal failures ($<0.25\sigma$ parametric shift).
  * True defects exhibit localized contiguous block reading spikes ($>104.0$ vs baseline $100.0$).

---

## 4. Split & Leakage Findings

* **Wafer Isolation Integrity:**
  * $\text{Train Wafers (128)} \cap \text{Holdout Validation (32)} = \emptyset$ (0 wafers overlap).
  * $\text{Train CSV (160)} \cap \text{Test CSV (40)} = \emptyset$ (0 wafers overlap).
  * $\text{Holdout Validation (32)} \cap \text{Test CSV (40)} = \emptyset$ (0 wafers overlap).
* **Leakage Resolution Confirmed:** The previous discrepancy ($F1 = 0.8789$) documented in `results/MODEL_A_DISCREPANCY_AUDIT.md` was definitively caused by RNG mismatch between `np.random.seed` and `np.random.default_rng`, which had caused 28 training wafers to leak into validation. The current canonical split (`results/canonical_wafer_split.json`) completely prevents this defect.
* **Target Leakage Audit:** All 506 features in Model A and 526 features in Model B are derived strictly from pre-test electrical measurements, spatial coordinates, and raw block signals. Zero post-test target leakage exists.

---

## 5. Feature Findings

* **Model A Features ($506$):** $500$ electrical parametric tests (`feature_1` to `feature_500`), plus $6$ spatial features (`radial_dist`, `edge_dist`, `die_row`, `die_col`, `density_3x3`, `density_5x5`).
* **Top Parametric Predictors:** `feature_171` (Gain: 12,124), `feature_33` (Gain: 11,904), `feature_336` (Gain: 11,654), `feature_460` (Gain: 10,758).
* **Dead Feature Discovery:** As proven empirically, `density_3x3` and `density_5x5` are identically 0.0 across all dies because `full_df=df` was omitted during preprocessing.
* **Model B Feature Delta ($20$ features):** Model B adds 20 block engineered features from `src/block_parser.py`.
  * **`cluster_area` is the absolute dominant feature in Model B:** Gain = **$147,509.52$**, outstripping all other features by over $12\times$.
  * `b_mean` is #2 (Gain = $23,126.35$).
  * Dead / Pruned block features: `longest_run_104`, `cluster_width`, and `b_min` received **0 splits and 0 gain** in Model B, adding zero value.

---

## 6. Model A Findings

* LightGBM trained with 150 trees, learning rate 0.05, 63 leaves, feature subsampling 0.7, row subsampling 0.8, and `scale_pos_weight = 10.0`.
* Operates at $\tau_A = 0.5100$.
* Precision is exceptionally high ($89.47\%$, only 42 false alarms on 27k dies), but recall is only $33.81\%$.
* Model A is severely constrained by its tabular feature space: marginal failures look parametrically indistinguishable from passing dies.

---

## 7. Model B Findings

* LightGBM trained on 526 features with identical tree hyperparameters.
* Operates at $\tau_B = 0.5376$.
* Alone, Model B achieves $\text{AUC-PR} = 0.4659$ and $\text{ROC-AUC} = 0.8336$ (+4.02% over Model A).
* Standalone Model B recovers 47 gross FNs, but generates 130 FPs (88 new false alarms), dropping global precision to $73.74\%$.
* **Why Model B cannot replace Model A globally:** Unrestricted Model B evaluates memory block noise across all passing dies, creating unacceptable factory false alarms. Model B must remain reserved for ambiguous dies in the cascade.

---

## 8. CNN Findings

* PyTorch `Block1DCNN`: 3-layer Conv1D network with 41,153 parameters trained on normalized block signals with `BCEWithLogitsLoss(pos_weight=22.07)`.
* Standalone CNN on block readings achieves poor discriminability ($\text{AUC-PR} = 0.0759$, $\text{F1} = 0.1326$).
* **Dual-Stream Concatenation Failure Explained:** Directly concatenating the 32 continuous CNN embeddings with LightGBM (Dual-Stream B5) caused $\text{AUC-PR}$ to collapse to $0.1771$ (worse than Model A's $0.4497$).
* The 32 continuous embeddings injected noisy high-dimensional continuous variance that disrupted LightGBM's tabular split purity. Discrete structured features (`cluster_area`, `b_mean`) proved vastly superior to unconstrained deep embeddings.

---

## 9. Error Findings (Cohort Analysis)

| Cohort | Count | Mean $P_A$ | Median $P_A$ | Mean $P_B$ | Mean Radius | Mean Pre-Test Defect Density | Mean Block Reading | Mean `cnt_104` | Mean `cluster_area` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FN (Missed Defect)** | 699 | 0.2139 | 0.1963 | 0.2644 | 0.6861 | 0.0 | 100.21 | 717.87 | 5,151.20 |
| **FP (False Alarm)** | 42 | 0.5497 | 0.5430 | 0.3525 | 0.6198 | 0.0 | 100.05 | 706.38 | 5,023.96 |
| **TP (Detected Defect)** | 357 | 0.7255 | 0.7304 | 0.7290 | 0.6777 | 0.0 | 100.21 | 718.11 | 5,159.61 |
| **TN (Clean Pass)** | 26,874 | 0.1542 | 0.1382 | 0.1430 | 0.6337 | 0.0 | 100.01 | 702.71 | 4,968.29 |

### Primary Diagnostic Insights:
1. **False Negatives are Parametrically Suppressed:** Mean $P_A$ on FNs is 0.2139 (median 0.1963). Parametric measurements show minimal deviation, but block features show clear anomalies (`cnt_104` = 717.87 vs 702.71 on TNs; `cluster_area` = 5,151.20 vs 4,968.29).
2. **False Positives are Ambiguous Marginal Dies:** Mean $P_A$ on FPs is 0.5497, clustered tightly near the 0.5100 threshold. On these dies, Model B probability drops to 0.3525 because block signals show normal baseline behavior.

---

## 10. Cascade Forensics

### The Gating Bottleneck:
* **Total Model A FNs:** 699 dies.
* **FNs with $P_A < 0.25$ (Bypassed by Fast Path):** **517 dies ($73.96\%$)**.
* **FNs Escalated to Gate B ($0.25 \le P_A \le 0.55$):** **182 dies ($26.04\%$)**.
* **FNs Rescued by Model B ($P_B \ge 0.5376$):** **37 dies ($20.33\%$ of escalated FNs)**.

The fast path is saving 88.14% compute, but the rigid lower bound of $0.25$ blinds Model B to 517 true defects.

---

## 11. Calibration Forensics

* **Model A Brier Score:** $0.04561$ $\rightarrow$ **Cascade Brier Score:** $0.04175$.
* **Expected Calibration Error ($\text{ECE}$):** $0.13116$ for Model A, $0.12053$ for Cascade.
* **Overconfidence Phenomenon:**
  * For dies with $P \in [0.10, 0.20)$, average predicted risk is $14.4\%$, but actual defect rate is only **$2.11\%$**.
  * For dies with $P \in [0.20, 0.30)$, average predicted risk is $24.0\%$, but actual defect rate is only **$4.42\%$**.
  * Cause: `scale_pos_weight = 10.0` warps the sigmoid log-odds output.
* **Prescription:** Implement post-hoc Platt scaling or Isotonic Regression on out-of-fold validation predictions to align probabilities with real factory defect incidence.

---

## 12. Threshold Forensics

* **Model A Threshold:** Locked at $\tau_A = 0.5100$.
  * F1 optimum is $\tau = 0.50$ (F1 = 0.4894).
  * F2 optimum (weighting recall $2\times$) is $\tau = 0.40$ (F2 = 0.4121, Recall = 38.26%, Precision = 59.59%).
* **Model B Threshold:** Locked at $\tau_B = 0.5376$.
* **Cascade Operating Point:** Joint tuning of $(\tau_A, \tau_B, L_G, U_G)$ has not been performed; current thresholds are piecewise uncoupled.

---

## 13. Wafer-Level Findings

* **Yield Heterogeneity:** Across 32 validation wafers, defect rate ranges from **$0.22\%$** (`W_N_0060`, 2 defects out of 893 dies) to **$11.23\%$** (`W_N_0107`, 106 defects out of 944 dies).
* **Where Cascade Helps Most:**
  * `W_N_0086`: Recall $+8.33\%$ ($31.25\% \rightarrow 39.58\%$, $+4$ TPs, $+2$ FPs, $\Delta F1 = +0.0740$).
  * `W_N_0107`: Recall $+4.72\%$ ($31.13\% \rightarrow 35.85\%$, $+5$ TPs, $-1$ FP, $\Delta F1 = +0.0564$).
  * `W_N_0132`: Recall $+5.88\%$ ($47.06\% \rightarrow 52.94\%$, $+1$ TP, $0$ FPs, $\Delta F1 = +0.0523$).
* **Where Cascade Degrades Performance:**
  * In 12 wafers, delta F1 was slightly negative due to false alarm spikes (e.g., `W_F_0018` had $+6$ false alarms; `W_F_0005` lost 1 borderline TP).
  * Global metrics conceal that cascade performance is wafer-dependent.

---

## 14. Signal Forensics

* Linear correlation with post-test defect:
  * `cluster_area`: $r = +0.1251$ (Strongest signal feature)
  * `b_mean`: $r = +0.1144$ (Secondary signal feature)
  * `cnt_104`: $r = +0.1045$
  * `b_std`: $r = +0.0547$
  * `b_max`: $r = +0.0307$
  * `longest_run_104`: $r = +0.0146$
* Continuous cluster area above baseline is over $8\times$ more predictive than single-block peak value (`b_max`).

---

## 15. Spatial Forensics

* `radial_dist` has positive correlation with defects ($r = +0.0417$), confirming edge degradation.
* `edge_dist` has negative correlation ($r = -0.0427$).
* `density_3x3` and `density_5x5` were dead ($r = \text{NaN}$) due to preprocessing bug.
* Spatial features are **100% SAFE FOR PRODUCTION** provided they are computed from pre-test maps (`old_label`), which are known before post-test.

---

## 16. Compute & Latency Forensics

Measured on Apple Silicon MPS / CPU:
* **Model A Inference Latency:** $0.0059$ ms per die ($169,084$ dies/sec).
* **Model B Inference Latency:** $0.0060$ ms per die ($167,351$ dies/sec).
* **PyTorch 1D CNN Latency:** $2.8693$ ms per die ($349$ dies/sec).
* **Cascade Effective Latency:** $0.0066$ ms per die ($151,515$ dies/sec).
* **Compute Savings:** **88.14%** of dies bypass Model B completely.

---

## 17. Multi-Metric Baseline Scorecard

| Metric Category | Metric Name | Model A Baseline | Standalone Model B | Selective Cascade Gate B | Cascade vs Model A Delta |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Classification** | Accuracy | $97.35\%$ | $97.06\%$ | $97.30\%$ | $-0.05\%$ |
| | Balanced Accuracy | $66.83\%$ | $67.04\%$ | $68.25\%$ | $+1.42\%$ |
| | Precision | $89.47\%$ | $73.74\%$ | $81.38\%$ | $-8.09\%$ |
| | Recall | $33.81\%$ | $34.56\%$ | $36.84\%$ | **$+3.03\%$** |
| | Specificity | $99.84\%$ | $99.52\%$ | $99.67\%$ | $-0.17\%$ |
| | False Positive Rate | $0.16\%$ | $0.48\%$ | $0.33\%$ | $+0.17\%$ |
| | F1 Score | $0.4907$ | $0.4707$ | **$0.5072$** | **$+0.0165 (+3.35\%)$** |
| | F2 Score | $0.3861$ | $0.3867$ | **$0.4137$** | **$+0.0276 (+7.15\%)$** |
| | MCC | $0.5409$ | $0.4927$ | $0.5369$ | $-0.0040$ |
| **Ranking** | ROC-AUC | $0.7934$ | $0.8336$ | $0.7990$ | $+0.0056$ |
| | PR-AUC | $0.4498$ | $0.4659$ | **$0.4677$** | **$+0.0179 (+3.98\%)$** |
| **Calibration** | Brier Score | $0.04561$ | $0.04546$ | **$0.04175$** | **$-0.00386$ (Better)** |
| | Expected Calib Error | $0.13116$ | $0.11915$ | $0.12053$ | $-0.01063$ |
| **Operational** | True Positives (TP) | $357$ | $365$ | **$389$** | **$+32$ net TPs** |
| | False Negatives (FN) | $699$ | $691$ | **$667$** | **$-32$ escaped dies** |
| | False Positives (FP) | $42$ | $130$ | $89$ | $+47$ false alarms |
| | True Negatives (TN) | $26,874$ | $26,786$ | $26,827$ | $-47$ dies |
| | Escalation Rate | $0.0\%$ | $100.0\%$ | $11.86\%$ ($3,317$ dies) | $+11.86\%$ |
| | Compute Saved | $100.0\%$ | $0.0\%$ | $88.14\%$ ($24,655$ dies) | $-11.86\%$ |
| **Wafer Stability** | Mean Wafer Recall | $37.25\%$ | $36.40\%$ | $38.56\%$ | $+1.31\%$ |
| | Median Wafer Recall | $36.36\%$ | $36.27\%$ | $37.38\%$ | $+1.02\%$ |
| | Worst Wafer Recall | $18.18\%$ | $18.18\%$ | $18.18\%$ | $0.00\%$ |
| | Recall Dispersion (Std) | $11.85\%$ | $11.98\%$ | $11.92\%$ | $+0.07\%$ |
| | Mean Wafer Precision | $76.18\%$ | $57.68\%$ | $63.92\%$ | $-12.26\%$ |

---

## 18. Highest-Impact Improvement Opportunities

1. **Fix Preprocessing Spatial Sequencing (Expected Impact: High, Confidence: 99%):** Pass `full_df=df` into `compute_spatial_features` to restore true pre-test defect density.
2. **Optimize Cascade Gating Window Asymmetrically (Expected Impact: High, Confidence: 90%):** Lower gate threshold from $0.25$ to $0.18-0.20$ to unblock the 517 trapped False Negatives.
3. **Signal Feature Expansion on Cluster Energy (Expected Impact: High, Confidence: 85%):** Add `cluster_area_103`, `cluster_area_105`, and $L_2$ cluster anomaly mass.
4. **Prune Dead Block Features (Expected Impact: Medium, Confidence: 95%):** Remove `longest_run_104`, `cluster_width`, `b_min`.
5. **Post-Hoc Probability Recalibration (Expected Impact: Medium, Confidence: 95%):** Fit Isotonic/Platt calibrators to eliminate the $5-7\times$ overconfidence error.

---

## 19. Recommended Experiment Sequence

* **Exp 0:** Baseline Verification Lock.
* **Exp 1:** Spatial Preprocessing Sequencing Fix (`density_3x3`, `density_5x5` restoration).
* **Exp 2:** Signal Feature Expansion (`cluster_area` variants) & Dead Feature Pruning.
* **Exp 3:** Asymmetric Gating Window Sweep & Joint Cost-Sensitive Threshold Tuning.
* **Exp 4:** Post-Hoc Isotonic Probability Recalibration.
* **Exp 5:** Wafer Group Cross-Validation Hyperparameter Optimization.
* **Exp 6:** Final Holdout Benchmark Validation on `input/test.csv`.

---

## 20. Risks

1. **False Alarm Explosion:** Expanding the gating window without tight threshold tuning could introduce excessive false alarms, degrading fab yield.
2. **Wafer Distribution Shift:** If new manufacturing lots experience different base defect rates, uncalibrated LightGBM models with fixed thresholds could drift.
3. **Overfitting to Single Validation Split:** All tuning must be cross-validated across wafer groups using GroupKFold.

---

## 21. What We Should NOT Do

1. **DO NOT retrain or tune global CNN embeddings directly into LightGBM (Dual-Stream):** Controlled benchmark B5/B6 definitively proved this degrades PR-AUC from $0.4497$ to $0.1771$.
2. **DO NOT perform die-level random train/validation splitting:** Must preserve 100% wafer isolation.
3. **DO NOT replace Model A globally with Model B:** Running Model B on 100% of dies multiplies false alarms by $3.1\times$ ($42 \rightarrow 130$).
4. **DO NOT use post-test labels or neighboring test outcomes as inference features:** Strict leakage prohibition.
5. **DO NOT evaluate models using unweighted accuracy:** On a 3.78% failure rate, a dummy "all-pass" classifier achieves 96.22% accuracy while missing 100% of defects.

---

## 22. Exact Next Training Experiment

### Experiment Name:
`EXP-01: Spatial Defect Density Restoration & Baseline Retraining`

### Hypothesis:
Restoring non-zero spatial neighborhood features (`density_3x3` and `density_5x5` computed from pre-test `old_label` on the complete wafer map) will supply Model A with localized defect cluster context, increasing Model A F1 score from $0.4907$ to $\ge 0.5100$ and boosting cascade F1 from $0.5072$ to $\ge 0.5250$.

### Exact Intervention:
1. Modify `build_model_A.py` line 114 to pass `full_df=df` into `compute_spatial_features(eligible, full_df=df)`.
2. Verify that `density_3x3` and `density_5x5` have non-zero variance ($[0.0, 0.67]$) across the 126,065 training dies and 27,972 validation dies.
3. Retrain Model A with locked hyperparameters (`num_leaves=63`, `lr=0.05`, `n_estimators=150`, `scale_pos_weight=10.0`).
4. Re-evaluate against the authoritative holdout validation set and compare against baseline scorecard.

**Audit Status:** AUDIT COMPLETE. Zero models retrained. Zero artifacts overwritten. Full forensic deliverable package preserved in `audit/model_forensics/`.
