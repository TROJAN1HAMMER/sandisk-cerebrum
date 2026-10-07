# Forensic Implementation Audit: Pre-Experiment Architectural & Infrastructure Analysis
**Platform**: SanDisk Cerebrum Die Yield Intelligence  
**Experiment Program**: EXP-05 through EXP-09 (Next-Gen Cerebrum Model Improvement)  
**Date**: October 2026  
**Status**: COMPLETED AUDIT PRIOR TO EXPERIMENTATION  

---

## 1. Current Architecture Overview

SanDisk Cerebrum is a two-stage selective multi-resolution cascade system engineered to predict post-burn-in latent die defects while minimizing computational overhead on high-density 3D NAND wafer production lines:

```
[Raw Die Arrival (Wafer Probe)]
               │
               ▼
   [Eligibility Filter: old_label == 0]
               │ (Rejects functional probe failures upfront)
               ▼
   [Stage 1: Model A (Coarse Resolution)]
   506 Features: 500 ATE electrical measurements + 6 spatial wafer features
   Inference: Fast LightGBM GBDT (τ_A = 0.5100)
               │
               ├────────────────────────┬────────────────────────┐
               ▼                        ▼                        ▼
       P_A < 0.295              0.295 ≤ P_A < 0.550              P_A ≥ 0.550
      [FAST PASS]             [SELECTIVE ESCALATION]             [FAST FAIL]
     Decision: PASS                 (6.22% of dies)            Decision: DEFECT
    (92.34% of dies)                    │                       (1.44% of dies)
                                        ▼
                           [Stage 2: Model B2 (Fine Res)]
                           536 Features: 506 Model A features +
                           20 Legacy Block + 10 Anomaly Block features
                           Inference: Fine LightGBM GBDT (τ_B = 0.5376)
                                        │
                                ┌───────┴───────┐
                                ▼               ▼
                           P_B < 0.5376    P_B ≥ 0.5376
                           Decision: PASS  Decision: DEFECT
```

### Auxiliary Forensic Signal Component
- **1D ConvNet (`Block1DCNN`, `cnn_1d_block_model.pt`)**: 3-layer 1D CNN with Adaptive Average Pooling and a 32-dim projection head.
- **Role in Current Production**: Diagnostic and sub-die attribution only (via occlusion / saliency maps in `src/explainability.py`). It is **deliberately NOT concatenated** into LightGBM feature vectors because prior empirical trials showed raw CNN embedding concatenation degraded PR-AUC from 0.4677 to 0.4190 due to distribution misalignment and gradient dilution.

---

## 2. Existing Model A Training Path

- **Script**: `build_model_A.py` and `src/model_a.py`.
- **Inputs**: 
  - 500 ATE continuous parametric measurements (`feature_1` .. `feature_500`).
  - 6 spatial context features (`radial_dist`, `density_3x3`, `density_5x5`, `edge_dist`, `die_row`, `die_col`).
- **Algorithm**: `lightgbm.LGBMClassifier`.
  - Hyperparameters: `n_estimators=150`, `learning_rate=0.05`, `num_leaves=31` (or 63 in tuning), `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=10.0`, `random_state=42`.
- **Artifacts**: Serialized to `models/model_A.pkl` and `models/model_A.txt`.
- **Locked Threshold**: $\tau_A = 0.5100$.

---

## 3. Existing Model B2 Training Path

- **Scripts**: `experiments/EXP-03/run_exp_03.py` and `experiments/EXP-03B/run_exp_03b.py`.
- **Inputs**: 536 tabular features total:
  - 506 Model A features (500 electrical + 6 restored spatial).
  - 20 legacy baseline block features (`b_mean`, `b_std`, `b_min`, `b_max`, `b_p90`, `b_p95`, `b_p99`, `max_minus_100`, `mad_100`, `cnt_103`, `cnt_104`, `cnt_105`, `max_roll50`, `max_roll100`, `max_roll50_dev`, `max_roll100_dev`, `longest_run_104`, `num_runs_104`, `cluster_width`, `cluster_area`).
  - 10 frozen candidate block anomaly features (`cnt_110`, `cnt_115`, `cnt_120`, `cluster_area_110`, `cluster_area_115`, `max_roll20`, `max_roll20_dev`, `max_roll20_rel`, `peak_prominence`, `top10_dev`).
- **Algorithm**: `lightgbm.LGBMClassifier`.
  - Hyperparameters: `n_estimators=150`, `learning_rate=0.05`, `num_leaves=63`, `subsample=0.8`, `colsample_bytree=0.7`, `scale_pos_weight=10.0`, `min_child_samples=20`, `random_state=42`.
- **Artifact**: `models/model_B.pkl` (mirrored from `experiments/EXP-03B/models/model_B_B2_confirmed.pkl`).
- **Locked Threshold**: $\tau_B = 0.5376$.

---

## 4. Existing CNN Path

- **Architecture**: `Block1DCNN` in `src/model_b.py`.
  - Layer 1: Conv1D ($k=15, s=2, p=7$, 16 filters) + BatchNorm1D + ReLU + MaxPool1D(2).
  - Layer 2: Conv1D ($k=15, s=2, p=7$, 32 filters) + BatchNorm1D + ReLU + MaxPool1D(2).
  - Layer 3: Conv1D ($k=15, s=1, p=7$, 64 filters) + BatchNorm1D + ReLU + AdaptiveAvgPool1D(1).
  - Embedding Head: Linear(64, 32) + BatchNorm1D + ReLU + Dropout(0.3).
  - Output Head: Linear(32, 1) -> BCEWithLogitsLoss with `pos_weight`.
- **Pre-processing**: Normalized centered around baseline 100.0: $(X - 100.0) / 15.0$.
- **Limitations**: Fixed receptive field, single dilation scale, lacks multi-scale temporal kernel resolution.

---

## 5. Existing Feature Extraction

1. **Spatial Features (`src/spatial.py`)**:
   - `radial_dist`: Normalized Euclidean distance of die $(r, c)$ from wafer center.
   - `edge_dist`: Normalized minimum distance to wafer bounding boundary.
   - `density_3x3` & `density_5x5`: Uniform spatial 2D filtering of `old_label` (pre-test electrical probe defect flag). Note: Strict audit ensures this uses ONLY `old_label` and wafer coordinates; NEVER post-test `label`.
2. **Block Features (`src/block_parser.py` & `extract_advanced_candidate_block_features`)**:
   - Parses space-delimited string of 2,000 float32 readings per die.
   - Purely intra-die: zero cross-die statistics or leakage.
   - Computes rolling windows (sizes 20, 50, 100), thresholds (103, 104, 105, 110, 115, 120), peak prominence, and cluster areas.

---

## 6. Existing Threshold Logic

- **Model A standalone threshold**: $\tau_A = 0.5100$.
- **Model B2 standalone threshold**: $\tau_B = 0.5376$.
- **Routing Gate Policy B (Preferred)**:
  - If $P_A < 0.295$: Fast Pass (assign final prediction = 0, final probability = $P_A$).
  - If $P_A \ge 0.550$: Fast Fail (assign final prediction = 1, final probability = $P_A$).
  - If $0.295 \le P_A < 0.550$: Escalate to Model B2 (assign final prediction = $(P_{B2} \ge 0.5376)$, final probability = $P_{B2}$).

---

## 7. Existing Validation Methodology

- **Wafer Partitioning**: Strict wafer-level grouped split. Dies from the same wafer NEVER appear across train and test splits to prevent spatial leakage.
  - Canonical Training Set (`input/train.csv`): 128 wafers, 126,065 eligible dies, 5,463 defects (4.33% defect rate).
  - Canonical Holdout Set (`results/canonical_wafer_split.json`): 32 wafers, 27,972 eligible dies, 1,056 defects (3.78% defect rate).
  - Independent Holdout Benchmark (`input/test.csv`, EXP-04): 40 unseen wafers, 32,598 eligible dies, 1,380 defects (4.23% defect rate).
- **Rule of Discipline**: The 40-wafer independent holdout is **strictly frozen** and was evaluated once in EXP-04. During all iterative experiments (EXP-05 through EXP-09), model training, tuning, and selection will be executed exclusively on training wafers using wafer-grouped K-fold cross-validation and canonical validation. Only the single final frozen candidate will receive a confirmation evaluation against the independent benchmark.

---

## 8. Existing Artifact & Versioning System

- **Authoritative Files**:
  - `results/AUTHORITATIVE_CLEAN_METRICS.json`: Canonical verification metrics.
  - `models/model_release.json`: Production release metadata for B2.
  - `models/model_A.pkl` / `models/model_B.pkl`: Production model binaries.
- **Immutable Rule**: All new experiments must output into isolated subdirectories under `experiments/` (`EXP-05_LATE_PROBABILITY_STACKING`, `EXP-06_MODEL_A_ENSEMBLE`, etc.) and preserve production model artifacts unmodified.

---

## 9. Integration Plan for Proposed Improvements

| Experiment | Proposed Improvement | Architecture Point | Strategy & Isolation |
| :--- | :--- | :--- | :--- |
| **EXP-05** | Late Probability Stacking | Stage 2 Fusion ($P_{\text{tabular}} \oplus P_{\text{signal}}$) | Investigate late fusion (linear blending, logit-space fusion, and wafer-grouped meta-classifier) combining tabular B2 probability with 1D CNN probability. Avoid raw embedding concatenation. |
| **EXP-06** | Model A Ensemble | Stage 1 Classifier ($506 \to P_A$) | Train LightGBM + XGBoost + CatBoost on 506 features using wafer-grouped CV. Test simple average, weighted blending, and evaluate both Model A standalone and through the full cascade. |
| **EXP-07** | Advanced Signal Features | Stage 2 Feature Matrix ($536 \to 536 + K$) | Extract compact, physically grounded Kurtosis/Skewness, Run-Length statistics across multiple thresholds, and FFT spectral descriptors. Perform ablation analysis. |
| **EXP-08** | Multi-Scale 1D Inception / Dilated TCN | Signal Feature Extraction / Auxiliary | Replace 3-layer CNN with Multi-Scale Inception / Dilated TCN. Test Mode A (forensic attribution) and Mode B (late probability fusion). |
| **EXP-09** | Isotonic Probability Calibration | Output Probability Calibration | Fit isotonic regression out-of-fold on the best candidate predictor. Evaluate Brier score, ECE, reliability curves, and decision thresholds. |

---

## 10. Potential Incompatibilities & Mitigation

1. **Embedding Concatenation Incompatibility**:
   - *Issue*: Directly concatenating CNN or TCN embeddings into LightGBM was proven in Phase 3C / EXP-03 to degrade PR-AUC.
   - *Mitigation*: Strictly enforce late probability-level fusion where signal models output scalar probabilities or log-odds before combination.
2. **Ensemble Inference Latency Overhead**:
   - *Issue*: Running LightGBM, XGBoost, and CatBoost on all dies in Stage 1 could triple Model A inference latency, undermining the high-throughput fast path (>10,000 dies/sec).
   - *Mitigation*: Benchmark inference latency and throughput per die. If Stage 1 latency increases excessively without a major PR-AUC/F2 lift, the ensemble cannot be promoted.
3. **High-Dimensional FFT Spectral Explosion**:
   - *Issue*: 2,000 FFT coefficients would add severe dimensionality and collinear noise to LightGBM.
   - *Mitigation*: Extract strictly compact, physically interpretable spectral summary statistics (e.g., dominant frequency, spectral energy, spectral centroid, spectral entropy, low/high band ratios) capped at 8-12 features.
4. **Target Leakage Risk in Signal Features**:
   - *Issue*: Temporal or frequency feature extraction must never use neighboring die information or downstream labels.
   - *Mitigation*: Feature extraction must remain strictly intra-die, operating solely on the single die's 2,000-block float array.
