# SanDisk Cerebrum — Final Model Technical Cheat Sheet

> **Quick-Reference Card for Hackathon Presentation & Technical Defense**  
> **Status:** Final Validated Candidate (`B2`) • Zero ML Optimization Remaining (Model Frozen)

---

## 1. System Parameters & Configuration

| Parameter | Value | Description / Provenance |
|---|---|---|
| **Model Family** | LightGBM Selective Cascade | Asymmetric two-stage cascade with confidence gating |
| **Active Release** | **B2 Enhanced Model B** | Frozen candidate promoted to production (`models/model_B.pkl`) |
| **Model A Architecture** | LightGBM Classifier | Coarse fab screening layer (EXP-01 restored spatial density) |
| **Model A Feature Count** | **506 features** | 500 electrical parametric + 6 wafer spatial coordinates/densities |
| **Model A Threshold ($\tau_A$)** | **0.5100** | Tuned to maximize screening precision while isolating boundary dies |
| **Cascade Uncertainty Gate** | **$[0.295, 0.550]$** | Asymmetric Gate B (EXP-02 search); bypasses 93.78% of dies |
| **Model B Architecture** | LightGBM Classifier | Fine sub-die anomaly classifier evaluated on escalated dies |
| **Model B Feature Count** | **536 features** | 500 electrical + 6 spatial + 20 legacy block + 10 candidate block |
| **Model B Threshold ($\tau_B$)** | **0.5376** | Frozen operating threshold from canonical holdout calibration |
| **CNN Auxiliary Engine** | 1D CNN PyTorch (`mps`/`cpu`) | **Attribution & Diagnostic only** (NOT concatenated into B2 features) |

---

## 2. Dataset Cohorts & Split Isolation

| Cohort | Wafers | Dies | Defect Count | Defect Rate | Role |
|---|---|---|---|---|---|
| **Fab Training Set** | 128 wafers | 111,888 dies | ~4,220 defects | 3.77% | Parameter learning (wafer-isolated) |
| **Canonical Holdout** | 32 wafers | 27,972 dies | 1,056 defects | 3.7752% | Historical baseline (B0) & EXP-01/02/03/03B |
| **Independent Ground Truth** | 40 wafers | 32,598 dies | 1,380 defects | 4.2334% | Final unbiased validation (EXP-04, `test.csv`) |

*Rule:* Zero wafer overlap across splits. No die-level random shuffling. No target leakage.

---

## 3. Authoritative Performance Matrix

| Metric | Historical Baseline (B0) | Confirmed B2 (Canonical) | **Final B2 (Independent EXP-04)** | 95% Bootstrap CI (EXP-04) |
|---|---|---|---|---|
| **Evaluation Population** | Canonical 32 Wafers | Canonical 32 Wafers | **Independent 40 Wafers** | 1,000 resamples |
| **Total Evaluated Dies** | 27,972 dies | 27,972 dies | **32,598 dies** | — |
| **Actual Defect Base** | 1,056 defects | 1,056 defects | **1,380 defects** | — |
| **Cascade F1 Score** | `0.5072` | `0.5254` | **`0.5177`** | `[0.4948, 0.5369]` |
| **Cascade F2 Score** | `0.4137` | `0.4213` | **`0.4190`** | `[0.3995, 0.4403]` |
| **Matthews Corr. Coeff (MCC)** | `0.5369` | `0.5674` | **`0.5517`** | `[0.5260, 0.5701]` |
| **Precision-Recall AUC (PR-AUC)** | `0.4677` | `0.4777` | **`0.4808`** | `[0.4465, 0.5089]` |
| **ROC-AUC** | `0.7990` | `0.7986` | **`0.8078`** | `[0.7932, 0.8220]` |
| **Precision** | `81.38%` | `89.32%` | **`85.22%`** | — |
| **Recall** | `36.84%` | `37.22%` | **`37.17%`** | — |
| **True Positives (TP)** | 389 | 393 | **513** | — |
| **False Negatives (FN)** | 667 | 663 | **867** | — |
| **False Positives (FP)** | 89 | 47 | **89** | — |
| **True Negatives (TN)** | 26,827 | 26,869 | **31,129** | — |
| **Escalation Rate to Model B** | `11.86%` | `5.98%` | **`6.22%` (2,028 dies)** | — |
| **Block Compute Saved** | `88.14%` | `94.02%` | **`93.78%` (30,570 dies)** | — |
| **Model A FN Rescued by B2** | — | 36 / 138 (26.09%) | **41 / 176 (23.30%)** | — |
| **Net Defect Gain vs Model A** | +32 | +36 | **+37 defects** (513 vs 476) | — |

---

## 4. Software & Platform Verifications

- **Unit & Integration Tests:** **38 / 38 PASS** (`python3 -m unittest discover tests`)
- **Frontend Production Build:** **PASS (0 errors)** (`tsc && vite build`)
- **Active Model B SHA-256:** `8877db7b55abcb8a7c28b9de6f662bb3dfacbbcfbdcacac6da0deeaf5314e09a`
- **Archival B0 Backup SHA-256:** `3f29d1d9b6ac1eb5b11bdc8db47238373eb219f67a86d2eff0007e6dbc58ef24`
- **Backend API Stack:** FastAPI 0.115, Uvicorn, Pandas 2.2, LightGBM 4.6, PyTorch 2.4 (MPS hardware acceleration)
- **Frontend Stack:** React 18, TypeScript 5.5, Vite 5.4, TailwindCSS, Lucide Icons, TanStack Query

---

## 5. Five Biggest Technical Improvements

1. **Spatial Defect Density Bug Fix (EXP-01):** Discovered that `density_3x3` and `density_5x5` were previously computed on an already-filtered subset, resulting in constant zero values. Restoring computation across the full wafer reticle restored pre-test physical spatial clustering.
2. **Selective Uncertainty Gating (EXP-02):** Replaced arbitrary symmetric thresholds with an asymmetric uncertainty band $[0.295, 0.550]$, slashing Model B escalations by ~50% while cutting compute overhead by **93.78%** without sacrificing recall.
3. **Multi-Threshold & Local Prominence Block Signals (EXP-03):** Replaced vulnerable legacy cluster measures with 10 robust anomaly features (`cnt_110`, `cnt_115`, `cnt_120`, `cluster_area_110/115`, `max_roll20`, `max_roll20_dev`, `max_roll20_rel`, `peak_prominence`, `top10_dev`) that differentiate true localized physical defects from baseline background drift.
4. **Independent Population Validation (EXP-04):** Verified the frozen candidate on 40 completely unobserved ground-truth wafers (32,598 dies). Proved that ranking discrimination generalizes stably: PR-AUC improved from `0.4777` to `0.4808` (+0.65%) and ROC-AUC improved from `0.7986` to `0.8078` (+1.16%).
5. **Net Defect Recovery Tradeoff:** Rescued 41 Model-A false negatives entering Model B (23.30% rescue rate), while 4 Model-A true positives were lost, yielding a net gain of +37 defectives captured (513 vs 476) at the cost of only +43 false alarms across 32,598 dies.

---

## 6. Five Biggest Remaining Limitations

1. **Bounded Absolute Recall (~37.17%):** While precision is high (85.22%), defect recall remains bounded under conservative operating points to prevent excessive wafer scrap costs.
2. **Probability Calibration Drift:** Raw GBDT probability outputs inside the ambiguous band $[0.295, 0.550]$ show minor uncalibrated compression; post-hoc isotonic or Platt scaling remains future work.
3. **CNN Embedding Degradation:** Concatenating deep CNN latent representations directly into LightGBM degraded tabular tree splits due to dimensionality mismatch; CNN is restricted to explainability/attribution.
4. **Static Fab Benchmark Horizon:** Evaluated on benchmark lots from a single fabrication node; real-world temporal sensor drift across tool PM cycles has not been modeled.
5. **Uniform Defect Loss Asymmetry:** The current operating point optimizes balanced F1/F2 rather than a dollar-denominated loss matrix specific to downstream packaging packaging costs vs scrap costs.
