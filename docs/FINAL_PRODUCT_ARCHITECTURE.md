# SanDisk Cerebrum: Final Product Architecture

## 1. System Mission & Core Paradigm

The SanDisk Cerebrum platform is an industrial-grade semiconductor yield-intelligence platform engineered for wafer fab operations, test engineering, and die quality assurance.

### The Five Core Yield Questions
1. **WHAT is happening?** Selective cascade defect detection screening dies with high precision.
2. **WHERE is it happening?** Reticle-level spatial intelligence partitioning wafers into physical hotspot clusters (Edge Bevel, Center Core, Defect Epicenter).
3. **WHY does Cerebrum believe it?** Multi-sensor evidence chains connecting parametric measurements, spatial neighborhood context, and 2,000-block waveform signals.
4. **HOW MUCH yield/risk is involved?** Wafer-level yield-at-risk indicators and Isotonic calibrated probabilities for manufacturing planning.
5. **WHAT should an engineer investigate next?** An interactive What-If policy simulator evaluating secondary analysis throughput budgets and defect recovery trade-offs.

---

## 2. Frozen Production Decision Engine

The production decision pipeline is strictly frozen to guarantee scientific reproducibility and validation integrity.

```
                    ┌───────────────────────────────┐
                    │ Raw Die Sensor & Spatial Data │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Model A Fast Screening    │
                    │   LightGBM (506 Features)     │
                    │         τ_A = 0.5100          │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Uncertainty Gating Window   │
                    │        [0.295, 0.550]         │
                    └───────┬───────────────┬───────┘
                            │               │
        P_A < 0.295 or P_A ≥ 0.550          │ 0.295 ≤ P_A < 0.550
                            │               │ (6.22% of dies)
                            ▼               ▼
                 ┌────────────────┐  ┌───────────────────────────────┐
                 │ FAST PATH PASS │  │    Model B2 Sub-Die Engine    │
                 │ 93.78% Compute │  │    LightGBM (536 Features)    │
                 │     Saved      │  │         τ_B = 0.5376          │
                 └────────────────┘  └───────────────┬───────────────┘
                                                     │
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │   Final Production Verdict    │
                                     │   +37 Net Defect Recovery     │
                                     └───────────────────────────────┘
```

### Frozen Production Specifications
- **Model A**: LightGBM GBDT classifier, 506 features (500 electrical + 6 pre-test spatial restoration features), $\tau_A = 0.5100$.
- **Gating Policy**: Asymmetric confidence gate $[0.295, 0.550]$. Dies outside this interval are resolved immediately on Stage 1.
- **Model B2**: LightGBM GBDT sub-die classifier, 536 features (506 base + 20 legacy block + 10 engineered block candidates: cnt_110/115/120, cluster_area, roll20, prominence, top10), $\tau_B = 0.5376$.

### Authoritative Independent Validation (EXP-04)
- **Population**: 40 unseen wafers, 32,598 independent dies, 1,380 true defects (4.23% defect base).
- **Precision**: 85.22% (89 False Positives).
- **Recall**: 37.17% (513 True Positives).
- **F1 Score**: 0.5177 (95% CI: [0.4948, 0.5369]).
- **F2 Score**: 0.4190.
- **MCC**: 0.5517.
- **PR-AUC / ROC-AUC**: 0.4808 / 0.8078.
- **Tester Escalation**: 6.22% (2,028 dies sent to secondary analysis).
- **Compute Saved**: 93.78% (30,570 fast-path bypass dies).
- **False Negative Rescue**: 41 / 176 (23.30% rescue rate).
- **Net TP Recovery**: +37 dies vs Model A standalone (41 Model-A FNs rescued, 4 Model-A TPs lost, net +37 TP).

---

## 3. Parallel Architecture Streams

Cerebrum enforces an architectural separation between operational decisioning, forensic waveform attribution, and calibrated risk reporting:

```
                            Input Sensor Data
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Stream 1:        │      │ Stream 2:        │      │ Stream 3:        │
│ Production GBDT  │      │ Forensic TCN     │      │ Isotonic Risk    │
│ Decision Engine  │      │ Attribution      │      │ Reporting        │
│                  │      │                  │      │                  │
│ • Model A        │      │ • 2,000 blocks   │      │ • Isotonic fit   │
│ • Gate B         │      │ • Receptive: 256 │      │ • ECE: 0.0035    │
│ • Model B2       │      │ • Saliency grad  │      │ • Yield-at-Risk  │
│ • Frozen verdict │      │ • Forensics only │      │ • Planning only  │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

1. **Production Decision Stream**: Executes the frozen LightGBM cascade. Determines physical pass/fail verdicts and fab tester routing.
2. **Forensic Waveform Stream**: Executes the PyTorch Multi-Scale Dilated Temporal Convolutional Network (receptive field 256, dilations $1, 2, 4, 8, 16$) to compute exact input-gradient saliency attributions across the 2,000-block signal. *Strictly forensic: never replaces or modifies production routing.*
3. **Risk Reporting Stream**: Applies the Isotonic Probability Calibrator (EXP-09, reducing ECE from 0.1250 to 0.0035) to present well-calibrated probabilities for yield planning and spatial exposure estimation without modifying operational decision thresholds.

---

## 4. Software Stack & Integration

- **Backend**: FastAPI (Python 3.14 on macOS MPS / CPU), Pydantic v2 schemas, NumPy, SciPy, LightGBM, PyTorch. 38/38 unit tests passing.
- **Frontend**: Vite 5 + React 18 + TypeScript + Tailwind CSS (Silicon Obsidian dark industrial theme), TanStack Query v5, Lucide React. 0 TypeScript errors, 0 build warnings.
- **API Interfaces**:
  - `GET /api/v1/system/health`, `GET /api/v1/system/status`
  - `GET /api/v1/models/metrics`, `GET /api/v1/models/benchmark`, `GET /api/v1/models/release`
  - `POST /api/v1/evaluation/simulate-policy` (in-memory policy simulator on 27,972 dies in <30ms)
  - `GET /api/v1/wafers`, `GET /api/v1/wafers/{wafer_id}/dies`, `GET /api/v1/wafers/{wafer_id}/hotspots`
  - `POST /api/v1/inference/die`, `POST /api/v1/inference/batch`
  - `GET /api/v1/inference/block-attribution`, `GET /api/v1/inference/tcn-forensics`
