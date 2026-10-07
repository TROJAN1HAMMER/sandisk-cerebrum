# INTEGRATION INVENTORY & REPOSITORY COMPARISON

**Target Platform:** SanDisk Cerebrum Die Yield Intelligence & Cascade Optimization Platform  
**Audit Scope:** Controlled Integration of `To_be_added/` into Production Cerebrum  
**Date:** October 7, 2026  
**Auditor:** Principal Release Integration Engineer  

---

## 1. System Inventory Summary

| Repository Area | Current System (Cerebrum Production) | Teammate System (`To_be_added/`) | Conflict Status | Integration Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **`src/block_parser.py`** | 20 legacy + 10 engineered B2 block features | Identical implementation | **None** (Identical) | Keep current. |
| **`src/cascade.py`** | Threshold sweep & cascade routing logic | Identical implementation | **None** (Identical) | Keep current. |
| **`src/data_loader.py`** | Wafer-grouped 128/32 split & loader | Identical implementation | **None** (Identical) | Keep current. |
| **`src/evaluation_protocol.py`** | Evaluation metrics, AUC-PR, Brier, ECE, bootstrap | Identical implementation | **None** (Identical) | Keep current. |
| **`src/spatial.py`** | 6 spatial features (`radial_dist`, `density_3x3`, `density_5x5`, `edge_dist`, coordinates) | Identical implementation | **None** (Identical) | Keep current. Enhanced spatial logic in candidate will be isolated in candidate feature module. |
| **`src/model_a.py`** | `ModelAWrapper` for LightGBM | Identical implementation | **None** (Identical) | Keep current production wrapper intact. |
| **`src/visualizer.py`** | Wafer plotting & confusion matrix renderers | Identical implementation | **None** (Identical) | Keep current. |
| **`src/explainability.py`** | Occlusion attribution + `MultiScaleDilatedTCN` saliency | Missing TCN gradient saliency function | **Subset** (Current is strictly superset) | Keep current to preserve offline TCN forensic tooling. |
| **`src/model_b.py`** | `Block1DCNN` + `MultiScaleDilatedTCN` architecture | Missing `MultiScaleDilatedTCN` | **Subset** (Current is strictly superset) | Keep current to preserve TCN architecture. |
| **`src/live_inference.py`** | Hardened production pipeline with locked `[0.295, 0.550]` fallback gate | Outdated fallback gate `[0.25, 0.55]` | **Regression in New** | Keep current production file; do NOT overwrite with outdated 0.25 gate. |
| **`models/cnn_1d_block_model.pt`** | PyTorch 1D CNN checkpoint (`0a081de...`) | Identical binary hash (`0a081de...`) | **None** (Identical) | Keep current. |
| **`models/model_A.pkl`** | LightGBM 506 features ($\tau_A=0.5100$) | Not provided as pkl (script only) | **N/A** | Keep current B2 production binary active. |
| **`models/model_B.pkl`** | LightGBM 536 features B2 ($\tau_B=0.5376$) | Not provided as pkl (script only) | **N/A** | Keep current B2 production binary active. |
| **`models/multiscale_tcn_model.pt`** | PyTorch TCN for offline forensics | Absent from `To_be_added` | **N/A** | Keep current. |
| **`models/isotonic_calibrator.pkl`**| Isotonic risk calibrator | Absent from `To_be_added` | **N/A** | Keep current. |
| **`experiments/`** | EXP-01 through EXP-10 + evaluation cache | 3 new scripts: `train_model_A_enhanced.py`, `train_model_B_enhanced.py`, `evaluate_enhanced_cascade.py` | **New Functionality** | Merge scripts into `experiments/` as candidate research pipeline. |
| **`backend/`** | 15 REST endpoints across 6 routers | Absent from `To_be_added` | **N/A** | Preserve 100% of existing backend contracts. |
| **`frontend/`** | Full React/TypeScript workstation | Absent from `To_be_added` | **N/A** | Preserve 100% of existing UI workstation. |
| **`input/`** | `train.csv` (3.6GB), `test.csv` (1.1GB) | `train.csv` (3.6GB identical) | **None** (Identical) | Keep current. |
| **`tests/`** | 38 integration & unit tests | Absent from `To_be_added` | **N/A** | Preserve 100% of test suite. |

---

## 2. Key Insights from Inventory

1. **`To_be_added/src/` is an older checkpoint of `src/`**:
   - 7 of the 10 files in `src/` are 100% byte-for-byte identical.
   - 2 files (`explainability.py`, `model_b.py`) are missing the TCN forensic code that we added for offline sequence attribution.
   - 1 file (`live_inference.py`) still has the obsolete `0.25` lower gate that we hardened to `0.295`.
   - **Conclusion:** Under NO circumstances should `To_be_added/src/` be copied over `src/`. Doing so would regress the hardened fallback gate and delete TCN forensic functionality!

2. **The Teammate's Contribution is Centered in `experiments/`**:
   - `train_model_A_enhanced.py`: Explores enhanced Model A features (polar angles $\sin\theta, \cos\theta$, $7\times 7$ density, PCA reconstruction error and statistical moments; serialized model has 517 features).
   - `train_model_B_enhanced.py`: Explores enhanced tabular Model B features (528 tabular features in serialized model) + late probability fusion with `Block1DCNN` (2,000 blocks as separate modality: $0.90 \times P_{\text{tabular}} + 0.10 \times P_{\text{CNN}}$).
   - `evaluate_enhanced_cascade.py`: Evaluates the candidate cascade on the **32 Canonical Holdout Wafers**.

3. **Validation Population Origin**:
   - The screenshot metrics were generated **strictly on the 32-wafer canonical holdout from `input/train.csv`** (1,056 defects), NOT on the 40-wafer independent EXP-04 test set (`input/test.csv`, 1,380 defects).
