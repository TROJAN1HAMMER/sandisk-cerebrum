# SanDisk Cerebrum: Selective Multi-Resolution Die Yield Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/DL-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Zero-Leakage Audit](https://img.shields.io/badge/Split%20Audit-VERIFIED%20CLEAN-3FB950.svg)](results/AUTHORITATIVE_CLEAN_METRICS.json)
[![Tests Passing](https://img.shields.io/badge/Unit%20Tests-14%2F14%20PASSING-success.svg)](tests/)

> **"Inspect every die cheaply. Escalate only uncertain dies for deeper sub-die block inspection."**

---

## Executive Summary

Modern semiconductor fabrication produces tens of thousands of dies per wafer. While coarse-grained parametric electrical testing is cheap and high-throughput, subtle localized defects often hide within microscopic sub-die block structures (2,000 sequence readings per die). Conversely, parsing sub-die sequences across 100% of manufactured dies introduces massive computational latency and false alarm noise on confident passes.

**SanDisk Cerebrum** solves this fundamental trade-off through a **Selective Multi-Resolution Cascade**:
1. **Model A (Screening Layer):** Evaluates $500$ parametric electrical features + $6$ spatial wafer context features across 100% of dies with ultra-low latency.
2. **Confidence Gating Window $[0.25, 0.55]$:** Immediately clears confident passes ($P_A < 0.25$) and flags blatant defects ($P_A > 0.55$) on the **Fast Path (88.14% of dies)**.
3. **Model B (Sub-Die Resolution Layer):** Invoked *strictly* for the **11.86% ambiguous dies**, extracting $20$ engineered anomaly features from $2,000$ sub-die readings to rescue missed failures.

---

## Benchmark & Authoritative Results

Evaluated on the **Canonical 32 Holdout Validation Wafers (27,972 dies, 0.0% train overlap)**:

| Metric | Model A Baseline | Standalone Model B (100%) | Selective Cascade Gate B (Recommended) | Selective Cascade Gate C (High Precision) |
| :--- | :---: | :---: | :---: | :---: |
| **Gating Window** | None (0% Escalation) | Unrestricted (100%) | **`[0.25, 0.55]`** | `[0.30, 0.60]` |
| **Operating Threshold** | `0.5100` | `0.5376` | **Dynamic (`0.5100` / `0.5376`)** | Dynamic (`0.5100` / `0.5376`) |
| **Fail F1 Score** | `0.4907` | `0.4707` | **`0.5072` (+3.35% rel. gain)** | `0.5064` (+3.20% rel. gain) |
| **AUC-PR** | `0.4498` | `0.4659` | **`0.4677` (+3.99% rel. gain)** | `0.4604` (+2.36% rel. gain) |
| **Precision** | `89.47%` | `73.74%` | **`81.38%`** | `87.07%` |
| **Recall** | `33.81%` | `34.56%` | **`36.84%`** | `35.70%` |
| **ROC-AUC** | `0.7934` | `0.8336` | **`0.7990`** | `0.7935` |
| **Escalation Count** | 0 dies (0.00%) | 27,972 dies (100.0%) | **`3,317 dies` (`11.86%`)** | `1,626 dies` (`5.81%`) |
| **Block Compute Saved**| 100.00% | 0.00% | **`88.14%` (24,655 dies fast-path)** | `94.19%` (26,346 dies fast-path) |
| **Gross FN Recovery** | — (0) | 47 / 699 (6.72%) | **`37 / 699` (`5.29%`)** | `30 / 699` (`4.29%`) |
| **Net TP Gain** | — (0) | +8 dies (365 vs 357) | **`+32 dies` (389 vs 357)** | `+20 dies` (377 vs 357) |
| **New False Positives**| — (0) | +88 dies (130 vs 42) | **`+47 dies` (89 vs 42)** | `+14 dies` (56 vs 42) |

---

## System Architecture

```
                                    [ 100% OF MANUFACTURED DIES (27,972) ]
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │    MODEL A FAST SCREENING    │
                                       │ 500 Parametric + 6 Spatial   │
                                       └──────────────┬───────────────┘
                                                      │
                                           Calculate Risk P_A
                                                      │
                         ┌────────────────────────────┼────────────────────────────┐
                         │                            │                            │
                         ▼                            ▼                            ▼
                 P_A < 0.25 (75.3%)           0.25 ≤ P_A ≤ 0.55 (11.9%)      P_A > 0.55 (12.8%)
                [ CONFIDENT PASS ]               [ UNCERTAIN ZONE ]          [ CONFIDENT FAIL ]
                         │                            │                            │
                         │                            ▼                            │
                         │             ┌──────────────────────────────┐            │
                         │             │    MODEL B SUB-DIE FUSION    │            │
                         │             │ 20 Block Anomaly Features    │            │
                         │             └──────────────┬───────────────┘            │
                         │                            │                            │
                         │                     P_B ≥ 0.5376                        │
                         │                       ┌────┴────┐                       │
                         │                       ▼         ▼                       │
                         │                     [PASS]    [FAIL]                    │
                         │                       │         │                       │
                         ▼                       ▼         ▼                       ▼
            ┌─────────────────────────┐                            ┌─────────────────────────┐
            │   CLEARED PASS (TN)     │                            │   FLAGGED DEFECT (TP)   │
            │   (88.14% Fast Path)    │                            │   (+32 Net TP Boost)    │
            └─────────────────────────┘                            └─────────────────────────┘
```

---

## Mathematical Resolution of Recovered Dies

Under **Cascade Gate B**, the confusion matrix decomposes as:
$$\text{Net TP Gain} = \text{Gross FNs Recovered} - \text{TPs Lost in Gate}$$
$$+32 = 37 - 5$$

* **Gross Model A FNs Recovered:** **37 dies** ($5.29\%$ of 699 missed fails rescued).
* **Model A TPs Lost:** **5 dies** ($1.4\%$ of 357 dies fell below 0.5376).
* **Net Additional True Positives Captured:** **+32 defects** ($389 \text{ TP}$ vs $357 \text{ TP}$).
* **Compute Savings:** **$88.14\%$** of validation dies never touch the block feature pipeline.

---

## Enterprise Control Room Dashboard

The project includes a 9-page Streamlit application:

1. **• Overview / Control Room:** Real-time production KPI cards, quantified gains, and pipeline architecture.
2. **• Wafer Intelligence:** Interactive Plotly wafer map with defect filtering, spatial coordinate mapping, and tooltips across all 32 holdout wafers.
3. **• Live Die Check:** Interactive live inference console with 3 modes (Validation Presets/Demos, CSV upload batch prediction, manual input) and 1D occlusion sensitivity visualization.
4. **• Model Lab:** Dedicated visual laboratory comparing Model A, Model B, and Gate Cascade with individual metric profiles, 2x2 confusion heatmaps, and classification donuts.
5. **• Model Comparison:** Judge-friendly comparative analytics with 6-dimension grouped bar charts, side-by-side confusion heatmaps, error population bar charts, and $A \rightarrow B$ delta distribution histograms.
6. **• Cascade Analytics:** Pareto trade-off curves ($F_1$ vs Escalation Rate), compute savings vs precision curves, and gating window sweep tables.
7. **• Model Performance:** Dynamic live evaluation engine (`▶ RUN FULL EVALUATION`) verifying reproducibility against the authoritative benchmark.
8. **• Error Analysis:** Confusion transition cards, recovery equations, and interactive population drill-down tables.
9. **• System Status:** Automated integrity badge board confirming canonical split integrity, zero-leakage compliance, and unit test results.

---

## Repository Structure

```
sandisk-cerebrum/
├── .gitignore                                 # Excludes raw >3.8GB CSVs while tracking all models & code
├── README.md                                  # Production enterprise documentation
├── requirements.txt                           # Frozen production dependencies
├── app.py                                     # 9-page Enterprise Streamlit Dashboard
├── audit_reproducibility.py                   # Automated verification & forensic audit script
├── build_model_A.py                           # Model A baseline training pipeline
├── build_model_B.py                           # Model B multi-resolution experiments & feature fusion
├── build_selective_cascade.py                 # Uncertainty gating sweep & cascade optimizer
├── predict.py                                 # Production batch inference CLI
├── config.yaml                                # Centralized threshold and gate configurations
│
├── src/                                       # Core modular libraries
│   ├── block_parser.py                        # Fast sub-die block parsing & 20 engineered features
│   ├── cascade.py                             # Uncertainty gate logic & cascade decision engine
│   ├── data_loader.py                         # Leakage-safe canonical wafer split loader
│   ├── evaluation_protocol.py                 # Standardized confusion matrix & ranking metrics
│   ├── explainability.py                      # 1D block occlusion attribution & feature importance
│   ├── live_inference.py                      # Real-time live inference & evaluation engine
│   ├── model_a.py                             # Model A inference wrapper
│   ├── model_b.py                             # Model B inference wrapper
│   ├── spatial.py                             # Wafer coordinate geometry & neighborhood densities
│   └── visualizer.py                          # Semiconductor plotting utilities
│
├── models/                                    # Production model checkpoints
│   ├── model_A.pkl / model_A.txt              # Trained LightGBM Model A (506 features)
│   ├── model_B.pkl                            # Trained LightGBM Model B (526 features)
│   ├── cnn_1d_block_model.pt                  # Trained 1D CNN PyTorch checkpoint
│   └── selective_cascade_meta.pkl             # Gating metadata
│
├── results/                                   # Authoritative benchmarks & audit reports
│   ├── AUTHORITATIVE_CLEAN_METRICS.json       # Single source of truth metrics
│   ├── canonical_wafer_split.json             # Canonical 128 train / 32 val wafer split
│   ├── final_clean_benchmark.csv              # Benchmark comparison table
│   ├── val_die_diagnostics.csv                # 3MB precomputed diagnostics for 27,972 dies
│   ├── clean_model_B_predictions.csv          # Clean validation prediction vectors
│   ├── runtime_evaluation_latest.json         # Dynamic live evaluation output
│   ├── MODEL_A_DISCREPANCY_AUDIT.md           # Forensic audit documentation
│   └── PHASE_3C_FINAL_SELECTION.md            # Final architectural selection report
│
├── tests/                                     # Automated test suites
│   ├── test_live_inference_and_evaluation.py  # Live inference, schema validation & eval tests
│   ├── test_phase3c_metrics.py                # Mathematical consistency & benchmark tests
│   └── test_split_integrity.py                # Zero-leakage canonical split verification
│
└── visualizations/                            # High-resolution publication plots
    ├── probability_calibration.png            # Reliability calibration curve
    ├── model_B_block_attribution.png          # 1D sub-die block occlusion sensitivity
    ├── model_B_delta_distribution.png         # Delta risk shift distribution
    └── wafer_prediction_map.png               # Spatial wafer map
```

---

## Quickstart & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/TROJAN1HAMMER/sandisk-cerebrum.git
cd sandisk-cerebrum

python3 -m venv .venv
source .venv/bin/activate
pip install -r "Hackathon Problem - Die Yield Prediction/requirements.txt"
```

### 2. Run Automated Unit Tests
```bash
cd "Hackathon Problem - Die Yield Prediction"
python3 -m unittest discover tests
```
*Expected output: `Ran 14 tests in 0.145s, OK`.*

### 3. Launch Enterprise Dashboard
```bash
streamlit run app.py --server.port 8501
```
Open **`http://localhost:8501`** in your browser.

---

## Leakage-Safe Audit Verification

All experimental partitions strictly adhere to the canonical wafer split:
* **Canonical Train Set:** 128 wafers ($126,065$ dies, $4.33\%$ failure rate).
* **Canonical Holdout Validation Set:** 32 wafers ($27,972$ dies, $3.78\%$ failure rate).
* **Wafer Overlap:** **0.0% (Zero Wafers)**.
* **Pre-Test Filter:** Strictly applied `old_label == 0` (only functional dies eligible for yield prediction).
* **Forbidden Features:** `wafer_id`, `old_label`, `label`, and post-test metadata are 100% blocked from all model inputs.

---

## Author & Attribution

* **Project:** SanDisk Cerebrum Die Yield Intelligence
* **Developer:** Balaachuanu (`TROJAN1HAMMER`)
* **Repository:** [https://github.com/TROJAN1HAMMER/sandisk-cerebrum](https://github.com/TROJAN1HAMMER/sandisk-cerebrum)
