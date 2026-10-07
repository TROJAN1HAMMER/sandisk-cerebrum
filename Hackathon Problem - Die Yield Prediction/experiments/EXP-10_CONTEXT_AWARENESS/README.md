# EXP-10: Context-Aware Die Intelligence

**Experiment:** EXP-10 Context-Aware Die Intelligence  
**Platform:** SanDisk Cerebrum Die Yield Prediction  
**Status:** Experimental Branch (Isolated from Production)  
**Baseline:** Frozen Production B2 Selective Cascade (506 Model A features, 536 Model B2 features)  
**Date:** October 2026  

---

## 1. Executive Charter & Objectives

The goal of EXP-10 is to test whether making the decision engine **context-aware** improves latent defect detection without degrading precision, compute efficiency, or wafer robustness.

Rather than adding hundreds of arbitrary features, EXP-10 provides the model with structured answers to four fundamental questions:
1. *"How abnormal is this die relative to its physical neighbours?"* (Family A: Local Neighbourhood)
2. *"What is the morphological topology of the defect cluster?"* (Family B: Spatial Topology)
3. *"How abnormal is the die relative to its own wafer baseline?"* (Family C: Wafer-Relative Context)
4. *"Where in the memory array is the 2,000-block analog disturbance located?"* (Family D: Signal Geometry)
5. *"Do the independent detection modalities agree with each other?"* (Family E: Evidence Concordance)

---

## 2. Experimental Architecture

```
                                  [Incoming Die: 500 Parametric + 6 Spatial]
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │   Stage 1: LightGBM Model A  │
                                       │       (506 features)         │
                                       └──────────────┬───────────────┘
                                                      │
                       ┌──────────────────────────────┼──────────────────────────────┐
                       │                              │                              │
                       ▼                              ▼                              ▼
                 P_A < 0.295                 0.295 ≤ P_A < 0.550                P_A ≥ 0.550
             [FAST-PATH CLEAR]               [SELECTIVE ESCALATION]           [HIGH-RISK DEFECT]
           (93.78% compute saved)                      │                      (Screening Confirmed)
                                                       ▼
                                       ┌──────────────────────────────┐
                                       │     Context Feature Engine   │
                                       │   B2 + Selected Context (C)  │
                                       └──────────────┬───────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │  Candidate Context Model B   │
                                       │    (LightGBM + Context)      │
                                       └──────────────┬───────────────┘
                                                      │
                                     P_B ≥ 0.5376     │    P_B < 0.5376
                                     ┌────────────────┴────────────────┐
                                     ▼                                 ▼
                             [RESCUED DEFECT]                   [RESTORED PASS]
```

---

## 3. Candidate Datasets & Controlled Variants

- **B0 (Control)**: Current frozen B2 feature set (536 features).
- **C1 (Neighbourhood)**: B2 + Family A: 7 physical neighborhood contrast features (543 features).
- **C2 (Spatial Topology)**: B2 + Family B: 6 directional pre-test density and morphology features (542 features).
- **C3 (Wafer-Relative)**: B2 + Family C: 8 intra-wafer z-scores and robust deviations (544 features).
- **C4 (Signal Geometry)**: B2 + Family D: 8 anomaly location and multi-scale geometry features (544 features).
- **C5 (Evidence Agreement)**: B2 + Family E: 5 cross-sensor interaction and concordance features (541 features).
- **C6 (All Context Families)**: B2 + Families A through E (570 features).
- **C7 (Strongest Selected)**: B2 + Top gain-ranked contextual descriptors selected evidence-driven from C6.

---

## 4. Key Artifacts in This Directory

- [`FEATURE_AUDIT.md`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/FEATURE_AUDIT.md): Detailed audit of existing 536 features vs missing context.
- [`FEATURE_DEFINITIONS.md`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/FEATURE_DEFINITIONS.md): Mathematical formulations for all 34 contextual features.
- [`FEATURE_LEAKAGE_AUDIT.md`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/FEATURE_LEAKAGE_AUDIT.md): Zero-leakage compliance audit.
- [`FEATURE_FAMILY_ABLATION.csv`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/FEATURE_FAMILY_ABLATION.csv): Complete metric delta table across all families.
- [`WAFER_ROBUSTNESS.csv`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/WAFER_ROBUSTNESS.csv): Per-wafer robustness breakdown across validation wafers.
- [`LATENCY_COMPARISON.csv`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/LATENCY_COMPARISON.csv): Inference latency and effective compute cost benchmark.
- [`METRICS.json`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/METRICS.json): Authoritative machine-readable validation and holdout metrics.
- [`RESULTS.md`](file:///Users/23MIS0012/Desktop/HACKS/SANDISK/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/Hackathon%20Problem%20-%20Die%20Yield%20Prediction/experiments/EXP-10_CONTEXT_AWARENESS/RESULTS.md): Detailed scientific breakdown and discussion.
