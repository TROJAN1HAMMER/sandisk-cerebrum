# SanDisk Cerebrum Architecture Evolution: EXP-01 to EXP-09

## 1. Evolution History

```
EXP-01: Spatial Defect Restoration
Restored pre-test spatial defect densities (3x3 and 5x5) from wafer probe old_label.
Lifted Model A PR-AUC from 0.4498 to 0.4609 (+2.47% rel).

          │
          ▼
EXP-02: Gating Window & Decision Space Mapping
Swept threshold surfaces and discovered Policy B Preferred gate [0.295, 0.550].
Cut escalation from 11.86% to 5.98% while preserving 94% compute savings.

          │
          ▼
EXP-03 / EXP-03B: Sub-Die Block Anomaly Engineering (Frozen B2)
Engineered 10 high-resolution sub-die block features (cnt_110, cluster_area_110, max_roll20).
Lifted Model B PR-AUC to 0.4779 and Cascade F1 to 0.5254 (confirmed 100% deterministic).

          │
          ▼
EXP-04: Frozen Independent Holdout Validation (input/test.csv)
First independent validation across 40 unseen wafers (32,598 dies, 1,380 defects).
Certified F1 = 0.5177, PR-AUC = 0.4808, 93.78% compute saved, Net TP Gain = +37.

          │
          ▼
EXP-05 through EXP-09: Next-Gen Improvement Program
Rigorous controlled evaluation of five candidate improvements:
  - EXP-05 Late Probability Stacking: REJECTED (Causes FP inflation, degrades PR-AUC to 0.4657)
  - EXP-06 Model A 3-Learner Ensemble: REJECTED (3.1x inference cost, drops compute saved from 94% to 74%)
  - EXP-07 Advanced Signal Features: REJECTED (Statistical, Run-Length, FFT dilute LightGBM tree splits)
  - EXP-08 Multi-Scale Dilated TCN: ADOPTED FOR FORENSICS (TCN achieves PR-AUC 0.0874 vs CNN 0.0794, but rejected for cascade decisioning)
  - EXP-09 Isotonic Calibration: ADOPTED FOR PROBABILITY RELIABILITY (-97% ECE, -43% Brier score)
```

## 2. Multi-Objective Architectural Assessment

| Architecture Stage | Primary Discrimination (PR-AUC / Recall) | Calibration (ECE / Brier) | Inference Cost (ms/die) | Compute Saved | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A Baseline** | 0.4498 / 33.8% | 0.129 / 0.045 | 0.0066 ms | 100% | Pre-screening filter |
| **Current B2 Cascade** | **0.4808 / 37.2%** | 0.121 / 0.044 | **0.0071 ms** | **93.78%** | **PRODUCTION CANONICAL** |
| **Candidate + TCN Attribution** | **0.4808 / 37.2%** | 0.121 / 0.044 | 0.0071 ms (opt-in 0.03ms) | 93.78% | **ADOPTED DIAGNOSTIC HEAD** |
| **Candidate + Isotonic Calibration**| **0.4808 / 37.2%** | **0.003 / 0.024** | 0.0071 ms | 93.78% | **ADOPTED RELIABILITY LAYER** |
