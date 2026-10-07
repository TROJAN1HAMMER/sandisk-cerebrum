# Final Repository Cleanup Audit — SanDisk Cerebrum

> **Document Type:** Authoritative Repository Cleanup & Hygiene Audit  
> **Release Target:** Unified Production Standard (`B2`) with Staged Candidate (`B3-DualStream`)  
> **Audit Status:** Pre-Cleanup & Post-Cleanup Verification  
> **Guiding Principle:** Safe cleanup only. Zero ML logic alterations. Zero metric modifications. Zero loss of scientific provenance.

---

## 1. Inventory & Disposition Matrix

| File / Directory | Size | Status | Reason & Provenance | Action |
|---|---|---|---|---|
| `models/model_A.pkl` | 1.1 MB | **KEEP** | Active B2 Production Model A (506 features, $\tau_A=0.5100$). | Preserve in place. |
| `models/model_B.pkl` | 1.1 MB | **KEEP** | Active B2 Production Model B (536 features, $\tau_B=0.5376$). | Preserve in place. |
| `models/model_release.json` | 3.5 KB | **KEEP** | Authoritative B2 release contract & metadata. | Preserve in place. |
| `models/candidates/model_A_enhanced.pkl` | 3.1 MB | **KEEP** | Staged B3 Candidate Model A (517 features, $\tau_A=0.5100$). | Preserve in place. |
| `models/candidates/model_B_tab_enhanced.pkl` | 3.4 MB | **KEEP** | Staged B3 Candidate Model B Tabular (528 features, $\tau_B=0.5200$). | Preserve in place. |
| `models/candidates/pca_pipeline.pkl` | 57 KB | **KEEP** | Frozen PCA projection pipeline ($k=25$) fitted on training passing dies. | Preserve in place. |
| `models/candidates/candidate_metadata.json`| 1.1 KB | **KEEP** | Checksums and artifact paths for staged B3 candidate. | Preserve in place. |
| `models/cnn_1d_block_model.pt` | 176 KB | **KEEP** | PyTorch 1D-CNN used for forensic attribution and B3 candidate late fusion. | Preserve in place. |
| `models/multiscale_tcn_model.pt` | 174 KB | **KEEP** | Forensic multi-scale dilated TCN model for sub-die waveform saliency. | Preserve in place. |
| `models/isotonic_calibrator.pkl` | 2.2 KB | **KEEP** | Calibrated probability mapping for fab planning and risk reporting. | Preserve in place. |
| `models/model_B_original_production_backup.pkl` | 1.1 MB | **KEEP** | Cryptographic rollback artifact for initial B0 release. | Preserve in place. |
| `models/model_A.txt` | 1.1 MB | **KEEP** | Human-readable LightGBM text dump for Model A verification. | Preserve in place. |
| `models/feature_metadata.json` | 43 KB | **KEEP** | Feature scaling and distribution statistics for 500 parametric features. | Preserve in place. |
| `models/selective_cascade_meta.pkl` | 1 KB | **KEEP** | Legacy metadata artifact for early cascade configuration. | Preserve in place. |
| `models/archive/` | 0 B | **DELETE** | Empty directory created during exploratory staging. | Delete directory. |
| `src/` (All 9 modules) | ~110 KB | **KEEP** | Core ML and feature extraction engine (both B2 and B3 candidate paths). | Preserve in place. |
| `backend/` (FastAPI stack) | ~40 KB | **KEEP** | REST API service, schemas, routes, and dependency container. | Preserve in place. |
| `frontend/` (React/Vite app) | ~500 KB | **KEEP** | Production yield intelligence user interface. | Preserve in place. |
| `tests/` (8 test suites) | ~45 KB | **KEEP** | Regression, unit, API, and split integrity test suites (39 tests). | Preserve in place. |
| `experiments/EXP-01` to `EXP-04` | ~1.5 MB | **KEEP** | Authoritative developmental progression establishing Production B2. | Preserve in place. |
| `experiments/EXP-05` to `EXP-10` | ~2.5 MB | **KEEP** | Optimization, forensic, calibration, and context-awareness experiment trails. | Preserve in place. |
| `experiments/teammate_candidate/` | ~38 KB | **KEEP** | Candidate evaluation and training scripts integrated from teammate. | Preserve in place. |
| `experiments/exp_cache/` | ~22 MB | **KEEP** | Cached block waveform tensors for rapid benchmark reproducibility. | Preserve in place. |
| `results/` | ~48 MB | **KEEP** | Historical ground-truth benchmarks, validation diagnostics, splits. | Preserve in place. |
| `audit/model_forensics/` | ~380 KB | **KEEP** | Initial red-team forensic audit records and baseline scorecards. | Preserve in place. |
| `input/` | ~3.6 GB | **KEEP** | Primary dataset partitions (`train.csv`, `test.csv`, `sample_submission.csv`). | Preserve in place. |
| `docs/` | ~350 KB | **KEEP** | Complete technical documentation, cheatsheets, and integration reports. | Preserve in place. |
| `test_predictions.csv` | 3.7 MB | **ARCHIVE** | Early Phase-3 submission predictions CSV. Not needed for runtime application. | Move to `archive/legacy/`. |
| `catboost_info/` | ~35 KB | **DELETE** | Temporary auto-generated CatBoost training logs from EXP-06. | Delete directory. |
| `To_be_added/` (Parallel repo) | 3.6 GB | **DELETE** | Redundant parallel codebase containing duplicate 3.6GB `train.csv`. Provenance scripts preserved in `experiments/teammate_candidate/` & `archive/teammate_source/`. | Archive scripts & delete. |

---

## 2. Directory Hygiene Actions

### Action 1: Create Dedicated Archive
- `archive/legacy/`: Houses offline submissions and legacy predictions not required at runtime (`test_predictions.csv`).
- `archive/teammate_source/`: Preserves original exploration scripts from the teammate's branch (`train_model_A_enhanced.py`, `train_model_B_enhanced.py`, `evaluate_enhanced_cascade.py`) for complete archival provenance.

### Action 2: Delete Confirmed Useless & Redundant Files
- Delete `catboost_info/` (temporary experiment logs).
- Delete `models/archive/` (empty placeholder).
- Delete `To_be_added/` (3.6 GB duplicate shadow codebase) after verifying zero runtime imports.

---

## 3. Dependency Verification

1. **Imports:** Grep across all `.py`, `.ts`, `.tsx`, `.json` confirms **0** active runtime imports or file references to `To_be_added`.
2. **API Routes:** All backend endpoints bind to `models/` and `models/candidates/`.
3. **Frontend Services:** All UI queries consume standardized REST endpoints (`/api/v1/...`).
4. **Data Isolation:** `input/` in the project root remains the sole ground-truth data store.
