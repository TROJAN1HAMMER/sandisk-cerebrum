# EXP-03: Model B Feature & Error-Profile Improvement

## Overview

**Experiment ID:** `EXP-03`  
**Date:** October 6, 2026  
**Status:** COMPLETE  
**Decision:** **PROMOTE CANDIDATE**  

This experiment investigates targeted feature engineering and error-profile optimization for the second-stage **Model B** classifier within the SanDisk Cerebrum selective cascade.

---

## Central Hypothesis Tested

> *Can Model B itself be made better at distinguishing subtle defective dies from healthy dies while preserving its strong ranking ability and keeping false alarms under control?*

**Result: CONFIRMED.** By pruning 4 dead baseline features (`longest_run_104`, `cluster_width`, `b_min`, `mad_100`), restoring genuine wafer-level spatial defect density (`density_3x3`, `density_5x5`), and introducing multi-scale high-threshold anomaly energy and local prominence features (`cnt_110`, `cnt_115`, `cnt_120`, `cluster_area_110`, `cluster_area_115`, `max_roll20`, `max_roll20_dev`, `max_roll20_rel`, `peak_prominence`, `top10_dev`):
* **Model B Standalone F1 jumped from 0.4707 to 0.4873 (+0.0166)**.
* **Model B Standalone PR-AUC jumped from 0.4658 to 0.4761 (+0.0103)**.
* **Model B Standalone False Positives plunged from 130 to 102 (-28 false alarms eliminated)**.
* **Cascade F1 reached 0.5254 (Variant B2) and 0.5243 (Variant B3)**.
* **Cascade False Positives plunged to 47–50 dies (down from 89 Production Baseline, -47.2% false alarms)**.
* **Inference latency remained 0.009 ms/die (>106,000 dies/sec)**.

---

## Directory Structure

```text
experiments/EXP-03/
├── README.md                          # Quick reference and summary
├── EXP-03_FINAL.md                    # Comprehensive scientific evaluation report
├── candidate_features.md              # Detailed physical definitions & leakage audit
├── run_exp_03.py                      # Deterministic Python experiment runner
├── hashes_before.json                 # Pre-experiment SHA-256 hashes of production artifacts
├── hashes_after.json                  # Post-experiment SHA-256 hashes (100% verified identical)
├── cached_block_features.npz          # Cached parsed block feature matrices
├── feature_forensics.csv              # Audit of baseline block feature gains, splits & cohort stats
├── cohort_statistics.csv              # Summary statistics across TP, FN, FP, TN cohorts
├── model_B_variants.csv               # Standalone metrics across B0, B1, B2, B3
├── model_B_feature_importance.csv     # Gain and split importances for all variants
├── cascade_variant_results.csv        # Complete cascade results under Policy A and Policy B
├── rescue_analysis.csv                # Ledger of Model A FNs rescued, lost, and false alarm cost
├── wafer_level_results.csv            # Detailed wafer-by-wafer breakdown across all 32 holdout wafers
├── inference_benchmark.csv            # Latency and throughput benchmarks
├── tau_B_sensitivity.csv              # Secondary sensitivity sweep across tau_B [0.50, 0.57]
├── models/
│   ├── model_B_B0.pkl                 # Exact production baseline model reproduction
│   ├── model_B_B1.pkl                 # Feature-pruned model artifact
│   ├── model_B_B2.pkl                 # Enhanced model artifact
│   └── model_B_B3.pkl                 # Optimal enhanced + pruned model artifact
└── plots/
    ├── model_B_pr_curves.png          # Standalone Precision-Recall curves across B0-B3
    ├── rescue_vs_false_alarms.png     # Rescue power vs false alarm cost
    └── feature_importance_B3.png      # Top feature gains in Variant B3
```

---

## Reproduction Command

```bash
python3 experiments/EXP-03/run_exp_03.py
```
