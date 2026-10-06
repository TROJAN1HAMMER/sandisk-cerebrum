# EXP-02: Joint Threshold & Asymmetric Cascade Optimization

## Overview

**Experiment ID:** `EXP-02`  
**Date:** October 6, 2026  
**Status:** COMPLETE  
**Decision:** **CONDITIONAL KEEP**  

This experiment performs a controlled scientific exploration of the **decision policy and operating point** of the multi-resolution cascade system combining the **EXP-01 spatially restored Model A** with the production **Model B** selective classifier.

---

## Key Question Answered

> *What Model A threshold and Gate B interval provide the best overall semiconductor defect-detection operating point while balancing recall, precision, F1/F2, MCC, false alarms, escaped defects, escalation rate, compute savings, and wafer-level robustness?*

---

## Directory Structure

```text
experiments/EXP-02/
├── README.md                          # Experiment overview and quick reference
├── EXP-02_FINAL.md                    # Comprehensive scientific evaluation report
├── run_exp_02.py                      # Deterministic Python experiment runner
├── hashes_before.json                 # Pre-experiment SHA-256 hashes of production artifacts
├── hashes_after.json                  # Post-experiment SHA-256 hashes (100% verified identical)
├── cached_val_data.npz                # Cached unrounded validation inference arrays
├── grid_results.csv                   # Full 961-point grid search (tau_A x gate_lower)
├── gate_upper_sweep.csv               # Secondary sweep over Gate B upper bound [0.50, 0.65]
├── pareto_frontier.csv                # 465 non-dominated multi-objective Pareto points
├── candidate_operating_points.csv     # Ranked candidates (PREFERRED, VIABLE, REJECT)
├── wafer_level_results.csv            # Detailed wafer-by-wafer breakdown across all 32 holdout wafers
├── recovery_ledger.csv                # Complete accounting of Model A dies sent, rescued, lost, and removed
├── trapped_fn_analysis.csv            # Investigation of 473 trapped Model A FNs across gate_lower
├── probability_band_analysis.csv      # Defect incidence and model behavior across probability bins
├── threshold_curves.csv               # Operating curves across tau_A
├── heatmap_f1.csv                     # 31x31 F1 response matrix
├── heatmap_f2.csv                     # 31x31 F2 response matrix
├── heatmap_recall.csv                 # 31x31 Recall response matrix
├── heatmap_mcc.csv                    # 31x31 MCC response matrix
├── heatmap_fp.csv                     # 31x31 False Positive response matrix
├── heatmap_fn.csv                     # 31x31 False Negative response matrix
├── heatmap_escalation.csv             # 31x31 Escalation rate response matrix
└── plots/
    ├── pareto_frontier.png            # Recall vs Precision Pareto curve with benchmarks
    ├── heatmap_f1_f2.png              # 2D surface response of Cascade F1 and F2
    ├── trapped_fn_recovery.png        # Defect rescue vs False alarm penalty across Gate B lower bound
    └── wafer_robustness.png           # Wafer-by-wafer F1 comparison across all 32 holdout wafers
```

---

## Reproduction Command

```bash
python3 experiments/EXP-02/run_exp_02.py
```

Reruns deterministically in ~5 seconds using cached validation arrays, reproducing all tables, CSVs, and plots bit-for-bit.
