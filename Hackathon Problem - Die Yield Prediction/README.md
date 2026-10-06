# SanDisk Cerebrum
## AI-Powered Die Yield Intelligence Platform

```text
================================================================================
                    FINAL VALIDATED CANDIDATE: B2 ENHANCED CASCADE
--------------------------------------------------------------------------------
 93.78% Compute Saved  │  0.5177 Cascade F1   │  0.4808 PR-AUC   │  0.8078 ROC-AUC
 37.17% Defect Recall  │  85.22% Precision    │  +37 Net TP Gain │  35/35 Tests Pass
--------------------------------------------------------------------------------
 Validated On: 40 Independent Wafers • 32,598 Dies • 1,380 Defects (input/test.csv)
================================================================================
```

**SanDisk Cerebrum** is an enterprise-grade semiconductor die yield intelligence system built for high-throughput fabrication facilities. By combining ultra-fast fab-level screening with an asymmetric uncertainty gate and sub-die anomaly intelligence, Cerebrum accurately detects subtle physical silicon failures while bypassing **93.78%** of healthy dies from expensive secondary test sequences.

---

## The Semiconductor Yield Problem

Modern high-density NAND flash and ASIC wafers contain tens of thousands of dies per wafer reticle. Manufacturing defect rates are inherently low (~3.7% to 4.2%), creating an extreme class-imbalance challenge:

1. **The Cost of Over-Testing:** Collecting and analyzing deep, 2,000-block sub-die electrical traces across 100% of manufactured dies creates severe automated test equipment (ATE) throughput bottlenecks, costing fab operators millions of dollars in test-floor latency.
2. **The Cost of Escaped Defects:** Parametric fab measurements alone miss localized microscopic defects (such as gate oxide breakdown or pinhole shorts), causing defective dies to ship to packaging where assembly costs compound tenfold.
3. **The False-Scrap Trap:** Naive machine learning models trained on raw sequences produce unacceptable false alarms on benign dies that merely exhibit wafer-wide baseline voltage drift, scrapping perfectly functional silicon.

---

## The Solution: Selective Multi-Resolution Cascade

SanDisk Cerebrum deploys a selective asymmetric cascade that matches inspection resolution to prediction confidence:

```mermaid
flowchart LR
    D[Die Telemetry] --> A[Model A<br/>506 Features<br/>Spatial-Restored GBDT]
    A --> G{Uncertainty Gate<br/>P_A in [0.295, 0.550]}

    G -->|P_A < 0.295| F[Fast Path Pass<br/>93.78% of Dies<br/>Bypass Model B]
    G -->|0.295 <= P_A < 0.550| B[B2 Enhanced Model B<br/>536 Features<br/>Sub-Die Anomaly Signals]
    G -->|P_A >= 0.550| ADEC[Model A Decision<br/>Confident Defect]

    B --> T{B2 Threshold<br/>tau_B = 0.5376}
    T -->|P_B < 0.5376| PASS[Pass]
    T -->|P_B >= 0.5376| FAIL[Defect]

    F --> OUT[Final Disposition]
    ADEC --> OUT
    PASS --> OUT
    FAIL --> OUT
```

1. **Stage 1 (Model A Screening):** Evaluates 100% of manufactured dies using 500 parametric electrical measurements and restored pre-test spatial neighborhood density in under 1.2 ms per die.
2. **Stage 2 (Uncertainty Gate $[0.295, 0.550]$):**
   - **Fast Path ($P_A < 0.295$):** 93.78% of dies are confidently healthy and bypass secondary testing entirely.
   - **Confident Defect ($P_A \ge 0.550$):** Flagged immediately as failures by Model A.
   - **Ambiguous Zone ($0.295 \le P_A < 0.550$):** Only 6.22% of boundary dies are escalated to Model B.
3. **Stage 3 (B2 Sub-Die Resolution):** Analyzes 2,000 sequential readings per die using 10 engineered anomaly features to rescue subtle defects while filtering out baseline drift.

---

## Key Results

Authoritative performance across populations, from historical baseline to independent validation:

| Metric | Historical Baseline (B0) | Confirmed B2 (Canonical) | **Final B2 (Independent EXP-04)** | 95% Bootstrap CI |
|---|---|---|---|---|
| **Population** | Canonical 32 Wafers | Canonical 32 Wafers | **Independent 40 Wafers** | 1,000 Resamples |
| **Dies / Defects** | 27,972 / 1,056 | 27,972 / 1,056 | **32,598 / 1,380** | — |
| **Defect Rate** | 3.7752% | 3.7752% | **4.2334%** | — |
| **Cascade F1** | `0.5072` | `0.5254` | **`0.5177`** | `[0.4948, 0.5369]` |
| **PR-AUC** | `0.4677` | `0.4777` | **`0.4808`** | `[0.4465, 0.5089]` |
| **ROC-AUC** | `0.7990` | `0.7986` | **`0.8078`** | `[0.7932, 0.8220]` |
| **Precision** | `81.38%` | `89.32%` | **`85.22%`** | — |
| **Recall** | `36.84%` | `37.22%` | **`37.17%`** | — |
| **True Positives (TP)** | 389 | 393 | **513** | — |
| **False Negatives (FN)**| 667 | 663 | **867** | — |
| **False Positives (FP)**| 89 | 47 | **89** | — |
| **True Negatives (TN)** | 26,827 | 26,869 | **31,129** | — |
| **Escalation Rate** | 11.86% (3,317 dies) | 5.98% (1,674 dies) | **`6.22%` (2,027 dies)** | — |
| **Compute Saved** | 88.14% | 94.02% | **`93.78%` (30,571 dies)**| — |
| **Net Defect Gain** | +32 vs Model A | +36 vs Model A | **`+37` vs Model A** | — |

---

## Why the Cascade?

* **Asymmetric Cost of Computation:** Parsing raw waveforms for 32,598 dies requires significant memory and CPU/GPU time. Bypassing 93.78% of dies reduces automated test overhead by over 15x.
* **Intelligent Precision Filtering:** Unconstrained sub-die models run on 100% of dies generate high absolute numbers of false alarms. Confining Model B strictly to the ambiguous zone $[0.295, 0.550]$ preserves an **85.22% precision**.
* **Defect Recovery Tradeoff:** On the independent 40-wafer test set, B2 rescued **41 out of 176** Model-A false negatives that entered the gate (**23.30% rescue rate**), producing a net gain of +37 defectives captured (513 vs 476) at the expense of only 43 additional false positives across 32,598 dies.

---

## Model Evolution (B0 → B2)

The system evolved through five rigorous experimental cycles documented in [docs/MODEL_EVOLUTION_REPORT.md](docs/MODEL_EVOLUTION_REPORT.md):

```text
BASELINE (B0)
      │  F1: 0.5072 | Gate: [0.25, 0.55] | Compute Saved: 88.14%
      ▼
EXP-01: Spatial Defect Density Restoration
      │  Discovered density_3x3 / 5x5 were zeroed due to pre-filtering bug.
      │  Restored wafer-wide reticle convolution. PR-AUC boosted +2.98%.
      ▼
EXP-02: Joint Threshold & Asymmetric Cascade Optimization
      │  Pareto sweep revealed [0.25, 0.295] dies were overwhelmingly healthy.
      │  Shifted gate to [0.295, 0.550]; halved Model B calls; compute saved 94.02%.
      ▼
EXP-03: Model B Feature & Error-Profile Improvement
      │  Replaced brittle legacy threshold 104 with 10 local anomaly features.
      │  Engineered Candidate B2 (536 features) and B3 (532 features).
      ▼
EXP-03B: Frozen Candidate Confirmation
      │  Locked definitions, re-extracted deterministically; verified exact metrics.
      │  B2 confirmed superior precision (89.32% vs 88.74%); frozen for evaluation.
      ▼
EXP-04: Independent Holdout Generalization
      │  Evaluated on 40 untouched wafers (32,598 dies).
      │  PR-AUC: 0.4808 (+0.65%), ROC-AUC: 0.8078 (+1.16%), F1: 0.5177.
      ▼
FINAL PRODUCTION PROMOTION: B2 ENHANCED MODEL B
```

---

## B2 Feature Engineering

Candidate B2 uses **536 features** derived directly from the physical testing process:

```text
500 Parametric Electrical Features (Fab Sensor Telemetry)
  +
  6 Wafer Spatial Features (radial_dist, edge_dist, die_row, die_col, density_3x3, density_5x5)
  +
 20 Legacy Block Statistics (b_mean, b_std, b_max, mad_50, cnt_100, roll_mean_max, etc.)
  +
 10 Engineered Block Anomaly Candidates (cnt_110/115/120, cluster_area_110/115, roll20, prominence, top10)
  =
536 Total Features
```

### Physical Rationale for Enhanced Block Features
* **Multi-Threshold Spike Counts (`cnt_110`, `cnt_115`, `cnt_120`):** Measure extreme electrical breakdown events rather than moderate noise.
* **Multi-Threshold Cluster Areas (`cluster_area_110`, `cluster_area_115`):** Quantify the spatial breadth of high-energy defects.
* **Rolling-20 Deviations (`max_roll20`, `max_roll20_dev`, `max_roll20_rel`):** Isolate transient spikes relative to the die's own baseline, preventing wafer-wide DC level shifts from triggering false alarms.
* **Peak Prominence (`peak_prominence`):** Peak amplitude minus local baseline floor.
* **Top-10 Energy (`top10_dev`):** Concentrated defect energy in the 10 highest readings.

---

## Validation Methodology & Leakage Controls

* **Wafer-Level Isolation:** Splitting is performed strictly by wafer lot. Zero random die shuffling across splits ensures models never observe adjacent dies from the same physical wafer during training.
* **Pre-Test Spatial Context:** Defect density convolutions strictly use pre-test spatial layouts; no test outcomes or label information leak between neighbors.
* **Frozen Pre-Evaluation:** Model B weights, feature definitions, and thresholds ($\tau_A=0.5100, \tau_B=0.5376, \text{Gate}=[0.295, 0.550]$) were frozen before touching the 40-wafer independent dataset (`input/test.csv`).

---

## Explainability & Diagnostics Workstation

The platform integrates deep semiconductor explainability into its workstation:
1. **Interactive Circular Wafer Maps:** Full SVG reticle visualizer with spatial defect clusters, edge-exclusion zones, and radial distance overlays.
2. **Die-Level Cascade Trace:** Inspect exact routing decisions: Model A screening score $\to$ Gating classification $\to$ Model B sub-die invocation $\to$ Final disposition.
3. **2,000-Block Waveform Viewer:** Zoom into the raw sub-die test sequence with interactive threshold bands (104, 110, 115, 120 units).
4. **PyTorch 1D CNN Occlusion Attribution:** Sliding occlusion window attribution identifies the exact temporal block sequence that triggered an anomalous decision.

---

## Platform Architecture

```text
React 18 / TypeScript / Vite Frontend (TailwindCSS, Lucide, TanStack Query)
                             │
                             ▼  HTTP / REST
FastAPI Backend (Uvicorn, Pydantic v2, CORS, State Lifecycle)
                             │
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
LightGBM Engine         PyTorch 1D CNN         Inference Pipeline
(Model A + B2)       (Attribution/Diagnostics) (Live Single & Batch)
```

* **Frontend:** React 18, TypeScript 5.5, Vite 5.4, TailwindCSS, TanStack Query, Lucide Icons.
* **Backend:** FastAPI 0.115, Python 3.10+, LightGBM 4.6, PyTorch 2.4 (MPS hardware acceleration), Pandas 2.2, NumPy 2.2.
* **Artifact Integrity:** Centralized release metadata in [models/model_release.json](models/model_release.json); historical baseline preserved in [models/model_B_original_production_backup.pkl](models/model_B_original_production_backup.pkl).

---

## Repository Structure

```text
.
├── backend/                  # FastAPI production backend
│   ├── api/
│   │   ├── routes/           # REST endpoints (models, wafers, inference, system)
│   │   └── schemas/          # Pydantic v2 validation contracts
│   ├── config.py             # Centralized settings and frozen gate parameters
│   ├── deps.py               # ML model loader and memory state lifecycle
│   └── main.py               # FastAPI application entrypoint
├── docs/                     # Technical documentation & defense reports
│   ├── FINAL_MODEL_CHEATSHEET.md    # Compact hackathon presentation cheat sheet
│   └── MODEL_EVOLUTION_REPORT.md    # Exhaustive 22-section technical viva report
├── experiments/              # Immutable historical experiment archives
│   ├── EXP-01/               # Spatial defect density restoration
│   ├── EXP-02/               # Joint threshold & gate optimization
│   ├── EXP-03/               # Model B feature engineering (B2 & B3)
│   ├── EXP-03B/              # Deterministic candidate confirmation
│   └── EXP-04/               # Final independent holdout generalization
├── frontend/                 # React 18 + Vite engineering workstation
│   ├── src/
│   │   ├── components/       # Silicon design system (WaferMap, MetricCards, etc.)
│   │   ├── pages/            # Workspaces (CommandCenter, WaferIntelligence, etc.)
│   │   └── services/         # Typed API clients
├── models/                   # Active and archival model artifacts
│   ├── model_A.pkl           # EXP-01 spatial-restored LightGBM (active)
│   ├── model_B.pkl           # B2 enhanced LightGBM (active production candidate)
│   ├── model_B_original_production_backup.pkl # B0 reference archival backup
│   ├── model_release.json    # Centralized model release metadata source of truth
│   └── cnn_1d_block_model.pt # PyTorch 1D CNN (diagnostics & attribution only)
├── results/                  # Authoritative benchmarks & validation datasets
│   ├── AUTHORITATIVE_CLEAN_METRICS.json
│   ├── canonical_wafer_split.json
│   └── val_die_diagnostics.csv
├── src/                      # Core ML engines and feature extractors
│   ├── live_inference.py     # End-to-end live cascade prediction pipeline
│   ├── explainability.py     # Occlusion attribution engine
│   └── block_parser.py       # Sequential 2,000-block feature engineering
└── tests/                    # Unit and integration test suite (35/35 passing)
```

---

## Running the Platform

### 1. Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd "Hackathon Problem - Die Yield Prediction"

# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Run Test Suite

```bash
# Execute all 35 backend tests (verifies B2 artifact, gates, endpoints, inference)
python3 -m unittest discover tests
```

### 3. Start Backend API Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

*Interactive Swagger API Docs:* `http://localhost:8000/docs`

### 4. Start Frontend Application

```bash
cd frontend
npm run dev
```

*Workstation UI:* `http://localhost:5173`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/system/health` | Service liveness health check |
| `GET` | `/api/v1/system/status` | Runtime model verification and active gates |
| `GET` | `/api/v1/models/release` | Centralized B2 release metadata & evaluation metrics |
| `GET` | `/api/v1/models/metrics` | Authoritative single source of truth metrics |
| `GET` | `/api/v1/models/benchmark` | Architecture comparison matrix & Pareto sweeps |
| `GET` | `/api/v1/wafers` | List all wafers with defect counts and yield statistics |
| `GET` | `/api/v1/wafers/{wafer_id}/dies` | Retrieve die coordinates, predictions, and status |
| `GET` | `/api/v1/wafers/{wafer_id}/dies/{row}/{col}` | Detailed single die diagnostic record |
| `POST`| `/api/v1/inference/die` | Real-time single die multi-resolution cascade inference |
| `POST`| `/api/v1/inference/batch` | High-throughput batch inference via CSV upload |
| `GET` | `/api/v1/inference/block-attribution`| PyTorch CNN occlusion attribution map |

---

## Testing & Build Verification

* **Unit & Integration Tests:** **`35/35 PASS`**
  ```bash
  python3 -m unittest discover tests
  # Ran 35 tests in 0.758s - OK
  ```
* **Frontend Production Build:** **`0 Errors`**
  ```bash
  cd frontend && npm run build
  # ✓ built in 3.67s
  ```

---

## Experiment Archives

Historical artifacts and detailed logs are preserved in:
* [experiments/EXP-01/EXP-01_FINAL.md](experiments/EXP-01/EXP-01_FINAL.md) — Spatial density restoration
* [experiments/EXP-02/EXP-02_FINAL.md](experiments/EXP-02/EXP-02_FINAL.md) — Operating point & gate optimization
* [experiments/EXP-03/EXP-03_FINAL.md](experiments/EXP-03/EXP-03_FINAL.md) — Feature engineering & B2 discovery
* [experiments/EXP-03B/EXP-03B_FINAL.md](experiments/EXP-03B/EXP-03B_FINAL.md) — Deterministic candidate confirmation
* [experiments/EXP-04/EXP-04_FINAL.md](experiments/EXP-04/EXP-04_FINAL.md) — Final independent generalization

---

## Limitations & Future Work

1. **Recall Operating Point:** B2 operates at **37.17% recall** to guarantee **85.22% precision**. Fabs prioritizing defect detection over scrap reduction can shift the operating point along the Pareto curve.
2. **Probability Calibration:** Decision scores inside $[0.295, 0.550]$ would benefit from isotonic post-calibration.
3. **Temporal Fab Drift:** Longitudinal validation across tool preventive maintenance (PM) cycles will be essential for ongoing model lifecycle management.

---

## Team & Hackathon Context

Developed for the **SanDisk / Western Digital AI Hackathon: Die Yield Prediction Challenge**.  
Architecture designed, validated, and finalized under zero data-leakage constraints.

---

## License

MIT License. Designed for semiconductor manufacturing research and yield intelligence applications.
