# System Instructions & Prerequisites Guide

## SanDisk Cerebrum: Selective Multi-Resolution Die Yield Intelligence

This document provides complete instructions on required external data files, target file paths, environment configuration, and end-to-end operational workflows for using and reproducing the entire system.

---

## 1. Required External Files & Target Placement

The repository contains all source code, pre-trained model checkpoints, test suites, and pre-computed evaluation results. To access the **full synthetic generation and scratch retraining pipeline**, one external dataset file is required.

| File Name | Description & Source | Target Destination Path | Mandatory For |
| :--- | :--- | :--- | :--- |
| **`LSWMD.pkl`** | **WM-811K Wafer Map Dataset**<br>Source: [Kaggle WM-811K Dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)<br>Format: Python Pickle (~811,000 wafer maps, ~3.5 GB uncompressed) | `Hackathon Problem - Die Yield Prediction/data/LSWMD.pkl` | Full raw data generation (`generate_data.py`) & training from scratch |
| **`train_data.csv`** | Generated training dataset (500 parametric features + 2,000 sub-die readings) | `Hackathon Problem - Die Yield Prediction/input/train_data.csv` | Re-training Model A, Model B, and 1D CNN |
| **`test_data.csv`** | Generated test dataset (40 wafers, ~35,000 dies) | `Hackathon Problem - Die Yield Prediction/input/test_data.csv` | Generating submission predictions |

> Note: If you only wish to run the interactive dashboard, inspect pre-computed results, or run inference with pre-trained models, you do **not** need to download `LSWMD.pkl`.

---

## 2. Environment Setup

### System Requirements
* **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS.
* **Python Version**: Python 3.10, 3.11, or 3.12 (Recommended).
* **RAM**: 8 GB minimum (16 GB recommended for full data generation).
* **GPU**: Optional (NVIDIA RTX with CUDA or CPU fallback).

### Installation Steps

1. **Clone the Repository and Navigate to the Project Root**:
   ```bash
   git clone https://github.com/TROJAN1HAMMER/sandisk-cerebrum.git
   cd sandisk-cerebrum
   ```

2. **Create and Activate a Virtual Environment**:
   * **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   * **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   cd "Hackathon Problem - Die Yield Prediction"
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install streamlit plotly torch lightgbm
   ```

---

## 3. Operational Modes & Execution Instructions

### Mode A: Launch Interactive Streamlit Control Room (Immediate Use)
No external dataset download is required. Uses pre-trained weights in `models/` and pre-computed diagnostics in `results/`.

```bash
cd "Hackathon Problem - Die Yield Prediction"
streamlit run app.py
```
* Access the web interface in your browser at `http://localhost:8501`.
* **Features**:
  * Page 1: Executive KPI Control Room & Quantified Cascade Impact.
  * Page 2: Spatial Wafer Map defect inspector across all 32 holdout validation wafers.
  * Page 3: Live Die Check (Interactive demo presets, single die scoring, CSV batch inference).
  * Page 4: Architectural Model Lab (Model A vs Model B vs Selective Cascade profiles).
  * Page 5: Head-to-Head Visual Comparison (Metrics, confusion heatmaps, error populations).
  * Page 6: Cascade Gating Analytics & Pareto Frontier trade-offs.
  * Page 7: Dynamic Full Live Evaluation against authoritative benchmarks.
  * Page 8: Error Analysis & Transition Dynamics.
  * Page 9: System Integrity, Canonical Split Verification & Artifact Registry.

---

### Mode B: Batch CLI Inference on New Wafer Data
To score arbitrary die test data using the pre-trained Selective Multi-Resolution Cascade:

```bash
cd "Hackathon Problem - Die Yield Prediction"
python predict.py --input test_predictions.csv --output results/my_scored_predictions.csv --lower_gate 0.30 --upper_gate 0.60
```

#### Input CSV Schema Requirements:
* `wafer_id` (string / int): Wafer identifier.
* `die_row`, `die_col` (int): Wafer matrix coordinates.
* `feature_1` to `feature_500` (float): Parametric electrical measurements.
* `block_readings` (string): 2,000 space-delimited float readings per die.
* `old_label` (optional, int): Pre-test functional filter (0 = passing eligible).
* `label` (optional, int): Post-test ground truth (if computing metrics).

---

### Mode C: Full Pipeline Reproduction From Scratch

Follow this sequence to regenerate data and train all models from scratch:

#### Step 1: Download Kaggle WM-811K
1. Download `LSWMD.pkl` from [Kaggle WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map).
2. Place `LSWMD.pkl` into `Hackathon Problem - Die Yield Prediction/data/LSWMD.pkl`.

#### Step 2: Generate Synthetic Wafer Datasets
```bash
cd "Hackathon Problem - Die Yield Prediction"
python generate_data.py --csv
```
* Generates `input/train_data.csv` (160 wafers, ~154,000 dies) and `input/test_data.csv` (40 wafers, ~35,000 dies).

#### Step 3: Train Model A (Parametric & Spatial Baseline)
```bash
python build_model_A.py
```
* Trains LightGBM on 506 features (500 parametric + 6 spatial context features).
* Saves `models/model_A.txt`, `models/model_A.pkl`, and `models/feature_metadata.json`.

#### Step 4: Train PyTorch 1D CNN & Build Model B
```bash
python build_model_B.py
```
* Trains the 3-layer `Block1DCNN` on 2,000 sub-die sequence measurements.
* Extracts 32-dimensional structural embeddings.
* Trains the Dual-Stream Multi-Resolution Model B fusion model.
* Saves `models/cnn_1d_block_model.pt` and `models/model_B.pkl`.

#### Step 5: Build Selective Uncertainty Cascade & Meta-Classifier
```bash
python build_selective_cascade.py
```
* Optimizes uncertainty gating thresholds (`[0.25, 0.55]` for Gate B, `[0.30, 0.60]` for Gate C).
* Fits the calibrated Logistic Regression meta-classifier for ambiguous dies.
* Saves `models/selective_cascade_meta.pkl` and validation benchmark metrics to `results/`.

#### Step 6: Generate Final Competition Submission Artifacts
```bash
python create_submission_artifacts.py
```
* Evaluates test holdout wafers and exports `test_predictions.csv`.

---

## 4. Audit & Quality Assurance Verification

Run the automated test and audit suite to verify zero-leakage compliance and reproducibility:

```bash
cd "Hackathon Problem - Die Yield Prediction"

# 1. Run Leakage and Split Integrity Unit Tests
python -m unittest discover -s tests -p "test_*.py"

# 2. Run Forensic Reproducibility Audit Script
python audit_reproducibility.py
```

Expected verification outcomes:
* `[PASS]`: Canonical 128 train / 32 validation wafer split overlap = 0.0% (Zero leakage).
* `[PASS]`: All evaluated validation dies adhere strictly to `old_label == 0`.
* `[PASS]`: Validation F1 score = 0.5072 (Cascade Gate B) with 88.14% compute reduction.
