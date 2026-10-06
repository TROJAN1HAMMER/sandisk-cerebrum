# EXP-03B: Independent Revalidation & Candidate Confirmation

## Overview
**EXP-03B** is the strict confirmation and candidate validation experiment for the SanDisk Cerebrum Die Yield Prediction platform. 

Following the exploratory feature discovery and candidate modeling performed in **EXP-03**, this experiment freezes all feature formulas, hyperparameters, decision thresholds, and split partitions to deterministically evaluate whether the reported improvements hold without validation-driven tuning or model-selection freedom.

## Core Principles & Controls
1. **Zero Feature Discovery**: Feature definitions from EXP-03 (`cnt_110`, `cnt_115`, `cnt_120`, `cluster_area_110`, `cluster_area_115`, `max_roll20`, `max_roll20_dev`, `max_roll20_rel`, `peak_prominence`, `top10_dev`, `density_3x3_restored`, `density_5x5_restored`) and pruning decisions (`longest_run_104`, `cluster_width`, `b_min`, `mad_100`) are 100% frozen.
2. **Locked Split**: The canonical 128-wafer training / 32-wafer holdout partition (`results/canonical_wafer_split.json`) is maintained with zero modifications.
3. **Locked Models & Thresholds**:
   - Model A: Spatial restored LightGBM checkpoint locked from EXP-01 ($\tau_A = 0.5100$).
   - Model B Operating Threshold: Frozen at $\tau_B = 0.5376$.
   - Policies Evaluated:
     - Policy A (EXP-01 Baseline Gate): Gate B = $[0.25, 0.55]$
     - Policy B (EXP-02 Preferred Gate): Gate B = $[0.295, 0.55]$
4. **Deterministic Bit-for-Bit Verification**: Candidates B2 and B3 were trained across multiple runs with fixed random seed (42). Predictions across runs were verified to be 100% identical.
5. **Zero Production Mutation**: Pre- and post-experiment SHA-256 hashes of all 9 production and benchmark artifacts verified byte-for-byte identical.

## Candidates Under Evaluation
- **B0**: Production Baseline Model B (526 features).
- **B2 (Enhanced)**: Model B with full engineered feature set (536 features: baseline + 10 block features + 2 restored spatial densities - 2 corrupted densities).
- **B3 (Enhanced + Pruned)**: Model B with enhanced features minus 4 low-utility/redundant features (`longest_run_104`, `cluster_width`, `b_min`, `mad_100`) (532 features).

## Primary Confirmation Results (Policy B Preferred: Gate B = [0.295, 0.55])
| Candidate | F1 | F2 | Precision | Recall | MCC | PR-AUC | ROC-AUC | TP | FN | FP | TN | Compute Saved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (Prod)** | 0.5190 | 0.4175 | 87.25% | 36.93% | 0.5581 | 0.4733 | 0.7971 | 390 | 666 | 57 | 26,859 | 94.02% |
| **B2 (Confirmed Leader)** | **0.5254** | **0.4213** | **89.32%** | **37.22%** | **0.5674** | **0.4777** | **0.7986** | **393** | **663** | **47** | **26,869** | **94.02%** |
| **B3 (Pruned)** | 0.5243 | 0.4210 | 88.71% | 37.22% | 0.5653 | 0.4726 | 0.7968 | 393 | 663 | 50 | 26,866 | 94.02% |

### Key Decision
- **Winner**: **B2 (Enhanced)**.
- **Rationale**: In the complete cascade, B2 dominates B3 across all primary metrics: higher F1 (+0.0011), higher F2 (+0.0003), higher MCC (+0.0021), higher PR-AUC (+0.0051), higher ROC-AUC (+0.0017), and lowest false alarm count (47 FP for B2 vs 50 FP for B3 vs 57 FP for B0). Furthermore, on the worst-performing wafer (`W_F_0018`), B2 improved F1 to 0.2222 with 3 false alarms, whereas B3 regressed to 0.1667 with 6 false alarms due to over-pruning.
- **Verdict**: **PROMOTE CANDIDATE** (Promotion grade candidate confirmed under frozen holdout; awaiting final evaluation on external/competition test set).

## Directory Structure
```text
experiments/EXP-03B/
├── README.md                              # Experiment overview and execution guide
├── run_exp_03b.py                         # Deterministic confirmation pipeline script
├── hashes_before.json                     # Pre-execution SHA-256 hashes of production files
├── hashes_after.json                      # Post-execution SHA-256 hashes verifying zero corruption
├── frozen_feature_manifest.json           # Frozen feature formulas, types, and schemas
├── model_B_confirmation.csv              # Standalone Model B evaluation metrics
├── cascade_confirmation.csv              # Complete cascade evaluation metrics (Policy A & B)
├── confirmation_summary.csv              # Primary decision summary table
├── rescue_confirmation.csv               # Rescue analysis of Model A false negatives & true positives
├── conditional_model_B.csv               # Model B quality restricted to escalated gate dies
├── wafer_level_confirmation.csv          # Per-wafer metrics across all 32 holdout wafers
├── model_stability.csv                   # Die-by-die prediction stability and shift analysis (B2 vs B3)
├── feature_importance_confirmation.csv   # Gain and split feature importances for B2 and B3
├── leakage_audit.md                      # Rigorous data leakage audit proof
├── future_experiments.md                 # Quarantined ideas for future development
├── models/
│   ├── model_B_B2_confirmed.pkl          # Serialized confirmed B2 model artifact
│   └── model_B_B3_confirmed.pkl          # Serialized confirmed B3 model artifact
├── plots/
│   ├── cascade_f1_comparison.png         # Bar chart comparing cascade F1 and FP
│   └── wafer_f1_distribution.png         # Boxplot of wafer-level F1 distributions
└── EXP-03B_FINAL.md                      # Comprehensive final scientific validation report
```
