# SanDisk Cerebrum — Master Authoritative Model Metrics Table

> **Document Type:** Project-Wide Authoritative Metrics & Provenance Table  
> **Repository:** SanDisk Cerebrum Die Yield Intelligence  
> **Release Standard:** B2 (Active Production Standard)  
> **Staged Candidate:** B3-DualStream (Validated Staged Candidate)  
> **Audit Status:** Verified against benchmark-producing serialized artifacts, JSON ledgers, and experiment reports. Zero synthetic or interpolated metrics.

---

## 1. Final Production Model

* **Model:** **B2**
* **Status:** **PRODUCTION**
* **Architecture:** LightGBM Selective Multi-Resolution Cascade (Model A: 506 features, Model B: 536 features)
* **Gating Policy:** Policy B Preferred ($\text{Gate} = [0.295, 0.550], \tau_A = 0.5100, \tau_B = 0.5376$)
* **Independent Validation Benchmark (`input/test.csv`):**
  * **Dies:** 32,598 eligible dies
  * **Wafers:** 40 completely unseen fabrication wafers
  * **Precision:** **85.22%**
  * **Recall:** **37.17%**
  * **F1:** **0.5177** (95% CI: $[0.4948, 0.5369]$)
  * **F2:** **0.4190**
  * **MCC:** **0.5517**
  * **PR-AUC:** **0.4808**
  * **ROC-AUC:** **0.8078**
  * **Accuracy:** **97.07%**
  * **Compute Saved:** **93.78%** (30,570 dies fast-path bypassed; 2,028 dies escalated)

---

## 2. Staged Candidate

* **Model:** **B3-DualStream**
* **Status:** **VALIDATED STAGED CANDIDATE**
* **Architecture:** Dual-Stream Cascade (Model A: 517 features, Model B: 528 tabular features + 1D-CNN late fusion on 2,000 blocks)
* **Gating Policy:** Candidate Standard ($\text{Gate} = [0.250, 0.550], \tau_A = 0.5100, \tau_B = 0.5200$)
* **Independent Validation Benchmark (`input/test.csv`):**
  * **Dies:** 32,598 eligible dies
  * **Wafers:** 40 completely unseen fabrication wafers
  * **Precision:** **87.67%**
  * **Recall:** **37.61%**
  * **F1:** **0.5264**
  * **F2:** **0.4246**
  * **MCC:** **0.5636**
  * **PR-AUC:** **0.5004**
  * **ROC-AUC:** **0.8270**
  * **Accuracy:** **97.14%**
  * **Compute Saved:** **96.05%** (31,310 dies fast-path bypassed; 1,288 dies escalated)

---

## 3. Master Authoritative Metrics Table

All values are transcribed directly from official experiment logs, evaluation JSON files, or release manifests. For B2 and B3, Canonical Holdout ($N=27,972$) and Independent Validation ($N=32,598$) are strictly reported as separate entries.

| Model / Experiment | Status | Dataset | Wafers | Dies | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 | F2 | MCC | PR-AUC | ROC-AUC | Compute Saved | Threshold / Gate |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **B0 Baseline (Original Prod)** | HISTORICAL REFERENCE | Canonical Holdout | 32 | 27,972 | 389 | 89 | 667 | 26,827 | 97.30% | 81.38% | 36.84% | 0.5072 | 0.4137 | 0.5369 | 0.4677 | 0.7990 | 88.14% | Gate: [0.250, 0.550], $\tau_A=0.5100, \tau_B=0.5376$ |
| **EXP-01 (Spatial Defect Density)** | EXPERIMENTAL (ACCEPTED) | Canonical Holdout | 32 | 27,972 | 394 | 83 | 662 | 26,833 | 97.34% | 82.60% | 37.31% | 0.5140 | 0.4191 | 0.5447 | 0.4755 | 0.7982 | 88.74% | Gate: [0.250, 0.550], $\tau_A=0.5100, \tau_B=0.5376$ |
| **EXP-02 (Joint Gate Optimization)** | EXPERIMENTAL (ACCEPTED) | Canonical Holdout | 32 | 27,972 | 391 | 57 | 665 | 26,859 | 97.42% | 87.28% | 37.03% | 0.5199 | 0.4185 | 0.5589 | 0.4741 | 0.7973 | 94.03% | Gate: [0.295, 0.550], $\tau_A=0.5100, \tau_B=0.5376$ |
| **EXP-03 (Model B Engineering - B2)** | EXPERIMENTAL (PROMOTED) | Canonical Holdout | 32 | 27,972 | 393 | 47 | 663 | 26,869 | 97.46% | 89.32% | 37.22% | 0.5254 | 0.4213 | 0.5674 | 0.4777 | 0.7986 | 94.02% | Gate: [0.295, 0.550], $\tau_A=0.5100, \tau_B=0.5376$ |
| **EXP-03B (Frozen Revalidation B2)** | VALIDATED CANDIDATE | Canonical Holdout | 32 | 27,972 | 393 | 47 | 663 | 26,869 | 97.46% | 89.32% | 37.22% | 0.5254 | 0.4213 | 0.5674 | 0.4777 | 0.7986 | 94.02% | Gate: [0.295, 0.550], $\tau_A=0.5100, \tau_B=0.5376$ |
| **EXP-04 (B2 Independent Test)** | VALIDATED CANDIDATE | Independent Test | 40 | 32,598 | 513 | 89 | 867 | 31,129 | 97.07% | 85.22% | 37.17% | 0.5177 | 0.4190 | 0.5517 | 0.4808 | 0.8078 | 93.78% | Gate: [0.295, 0.550], $\tau_A=0.5100, \tau_B=0.5376$ |
| **B2 FINAL PRODUCTION (Holdout)** | PRODUCTION (HOLDOUT REF) | Canonical Holdout | 32 | 27,972 | 393 | 47 | 663 | 26,869 | 97.46% | 89.32% | 37.22% | 0.5254 | 0.4213 | 0.5674 | 0.4777 | 0.7986 | 94.02% | Gate: [0.295, 0.550], $\tau_A=0.5100, \tau_B=0.5376$ |
| **B2 FINAL PRODUCTION (EXP-04)** | PRODUCTION | Independent Test | 40 | 32,598 | 513 | 89 | 867 | 31,129 | 97.07% | 85.22% | 37.17% | 0.5177 | 0.4190 | 0.5517 | 0.4808 | 0.8078 | 93.78% | Gate: [0.295, 0.550], $\tau_A=0.5100, \tau_B=0.5376$ |
| **EXP-05 (Late Prob Stacking - Blend)** | REJECTED | Canonical Holdout | 32 | 27,972 | 411 | 128 | 645 | 26,788 | 97.24% | 76.25% | 38.92% | 0.5154 | 0.4315 | 0.5330 | 0.4657 | 0.7942 | 94.03% | Gate: [0.295, 0.550], $\alpha=0.63, \tau_A=0.5100, \tau_B=0.5376$ |
| **EXP-06 (Tri-Model Ensemble)** | REJECTED | Canonical Holdout | 32 | 27,972 | 388 | 54 | 668 | 26,862 | 97.42% | 87.78% | 36.74% | 0.5180 | 0.4158 | 0.5585 | 0.4723 | 0.7991 | 93.42% | Gate: [0.295, 0.550], $\tau_A=0.5100, \tau_B=0.5376$ |
| **EXP-07 (Signal Waveform Features)** | REJECTED | Canonical Holdout | 32 | 27,972 | 389 | 57 | 667 | 26,859 | 97.41% | 87.22% | 36.84% | 0.5180 | 0.4165 | 0.5573 | 0.4726 | 0.7954 | 94.03% | Gate: [0.295, 0.550], $\tau_A=0.5100, \tau_B=0.5376$ (580 feats) |
| **EXP-08 (Multi-Scale Dilated TCN)** | EXPERIMENTAL (FORENSICS ONLY) | Canonical Holdout | 32 | 27,972 | 388 | 40 | 668 | 26,876 | 97.47% | 90.65% | 36.74% | 0.5229 | 0.4170 | 0.5682 | 0.4794 | 0.7992 | 94.03% | Gate: [0.295, 0.550], $\tau_A=0.5100, \tau_B=0.5376$ |
| **EXP-09 (Isotonic Risk Calibration)** | EXPERIMENTAL (ECE REPORTING) | Canonical Holdout | 32 | 27,972 | 396 | 66 | 660 | 26,850 | 97.40% | 85.71% | 37.50% | 0.5217 | 0.4225 | 0.5571 | 0.4799 | 0.8012 | 94.37% | Gate: [0.0588, 0.5120], $\tau_A=0.3424, \tau_B=0.3080$ |
| **EXP-10 (Context-Aware - C1 Holdout)**| REJECTED | Canonical Holdout | 32 | 27,972 | 396 | 54 | 660 | 26,862 | 97.45% | 88.00% | 37.50% | 0.5259 | 0.4226 | 0.5638 | 0.4750 | 0.7963 | 94.03% | Gate: [0.295, 0.550], $\tau_A=0.5100, \tau_B=0.5376$ (543 feats) |
| **EXP-10 (Context-Aware - C1 EXP-04)** | REJECTED | Independent Test | 40 | 32,598 | 510 | 87 | 870 | 31,131 | 97.06% | 85.43% | 36.96% | 0.5159 | 0.4169 | 0.5508 | 0.4803 | 0.8081 | 93.78% | Gate: [0.295, 0.550], $\tau_A=0.5100, \tau_B=0.5376$ |
| **B3-DualStream (Holdout Validation)** | VALIDATED CANDIDATE | Canonical Holdout | 32 | 27,972 | 413 | 92 | 643 | 26,824 | 97.37% | 81.78% | 39.11% | 0.5291 | N/A — not reported | N/A — not reported | 0.4967 | N/A — not reported | 94.68% | Gate: [0.250, 0.550], $\tau_A=0.5100, \tau_B=0.5200$ |
| **B3-DualStream (Independent EXP-04)**| VALIDATED STAGED CANDIDATE | Independent Test | 40 | 32,598 | 519 | 73 | 861 | 31,145 | 97.14% | 87.67% | 37.61% | 0.5264 | 0.4246 | 0.5636 | 0.5004 | 0.8270 | 96.05% | Gate: [0.250, 0.550], $\tau_A=0.5100, \tau_B=0.5200$ |
| **B3 High-Recall Policy (EXP-04)** | EXPERIMENTAL POLICY | Independent Test | 40 | 32,598 | 535 | 115 | 845 | 31,103 | 97.05% | 82.31% | 38.77% | 0.5271 | 0.4335 | 0.5531 | 0.5092 | 0.8291 | 92.63% | Gate: [0.200, 0.500], $\tau_A=0.5100, \tau_B=0.5000$ |

---

## 4. Standalone Model Components Reference

To provide complete technical provenance, the table below documents the individual component models (evaluated standalone across all eligible dies before cascading):

| Model Component | Status | Dataset | Wafers | Dies | TP | FP | FN | TN | Precision | Recall | F1 | PR-AUC | ROC-AUC | Threshold |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model A Baseline (B0)** | Historical | Canonical Holdout | 32 | 27,972 | 357 | 42 | 699 | 26,874 | 89.47% | 33.81% | 0.4907 | 0.4497 | 0.7934 | $\tau_A = 0.5100$ |
| **Model A EXP-01 (Spatial)** | Accepted | Canonical Holdout | 32 | 27,972 | 349 | 40 | 707 | 26,876 | 89.72% | 33.05% | 0.4830 | 0.4609 | 0.8016 | $\tau_A = 0.5100$ |
| **Model A B2 (Independent)**| Production | Independent Test | 40 | 32,598 | 476 | 46 | 904 | 31,172 | 91.19% | 34.49% | 0.5005 | 0.4662 | 0.8043 | $\tau_A = 0.5100$ |
| **Model B Baseline (B0)** | Historical | Canonical Holdout | 32 | 27,972 | 365 | 130 | 691 | 26,786 | 73.74% | 34.56% | 0.4707 | 0.4658 | 0.8336 | $\tau_B = 0.5376$ |
| **Model B B2 Enhanced (B2)**| Production | Canonical Holdout | 32 | 27,972 | 363 | 110 | 693 | 26,806 | 76.74% | 34.38% | 0.4748 | 0.4665 | 0.8297 | $\tau_B = 0.5376$ |
| **Model B B3 Pruned** | Rejected | Canonical Holdout | 32 | 27,972 | 373 | 102 | 683 | 26,814 | 78.53% | 35.32% | 0.4873 | 0.4761 | 0.8354 | $\tau_B = 0.5376$ |
| **Model B B2 (Independent)**| Production | Independent Test | 40 | 32,598 | 476 | 157 | 904 | 31,061 | 75.20% | 34.49% | 0.4729 | 0.4879 | 0.8458 | $\tau_B = 0.5376$ |
| **Candidate Model A (B3)** | Staged | Independent Test | 40 | 32,598 | 491 | 46 | 889 | 31,172 | 91.43% | 35.58% | 0.5123 | 0.4856 | 0.8250 | $\tau_A = 0.5100$ |
| **Candidate Model B (B3)** | Staged | Independent Test | 40 | 32,598 | 512 | 77 | 868 | 31,141 | 86.93% | 37.10% | 0.5201 | 0.5306 | 0.8653 | $\tau_B = 0.5200$ |

---

## 5. Source of Truth / Provenance Directory

Every number in the master tables traces back to an immutable project artifact:

| Experiment / Metric Group | Primary Authoritative Source Artifact | Path in Repository | Cryptographic / File Reference |
| :--- | :--- | :--- | :--- |
| **B0 Baseline (Canonical Holdout)** | Clean Benchmark Verification File | `results/AUTHORITATIVE_CLEAN_METRICS.json` | Hash: `569aaa6bdf388477aebb77e759fa9c04cd29d3c30ca70c3e6f7737544163547a` |
| **EXP-01 (Spatial Restoration)** | Experiment Report & Metric Manifest | `experiments/EXP-01/EXP-01_FINAL.md` | Model Checkpoint: `experiments/EXP-01/model_A_spatial.txt` |
| **EXP-02 (Gating Optimization)** | Grid Search & Policy Evaluation | `experiments/EXP-02/EXP-02_FINAL.md` | Grid Data: `experiments/EXP-02/grid_results.csv` |
| **EXP-03 (Model B Engineering)** | Variant Benchmark & Forensics | `experiments/EXP-03/EXP-03_FINAL.md` | Variant Summary: `experiments/EXP-03/cascade_variant_results.csv` |
| **EXP-03B (Frozen Revalidation)** | Confirmation Audit & Feature Manifest | `experiments/EXP-03B/EXP-03B_FINAL.md` | Manifest: `experiments/EXP-03B/frozen_feature_manifest.json` |
| **EXP-04 (B2 Independent Validation)**| Test Set Benchmark JSON | `experiments/EXP-04/independent_metrics.json` | Bootstrap CI: `experiments/EXP-04/bootstrap_results.json` |
| **B2 Active Production Metadata** | Release Registry Contract | `models/model_release.json` | Production Checkpoint: `models/model_B.pkl` |
| **B2 Latest Live Execution** | Live Evaluation JSON | `results/runtime_evaluation_latest.json` | Timestamp: `2026-10-07T18:17:20Z` |
| **EXP-05 (Late Prob Stacking)** | Stack Evaluation JSON & Report | `experiments/EXP-05_LATE_PROBABILITY_STACKING/METRICS.json` | `experiments/EXP-05_LATE_PROBABILITY_STACKING/RESULTS.md` |
| **EXP-06 (Tri-Model Ensemble)** | Ensemble Benchmark JSON | `experiments/EXP-06_MODEL_A_ENSEMBLE/METRICS.json` | `experiments/EXP-06_MODEL_A_ENSEMBLE/RESULTS.md` |
| **EXP-07 (Signal Features)** | 44 Feature Ablation JSON | `experiments/EXP-07_SIGNAL_FEATURES/METRICS.json` | `experiments/EXP-07_SIGNAL_FEATURES/RESULTS.md` |
| **EXP-08 (Multi-Scale Dilated TCN)** | TCN Sequence Benchmark JSON | `experiments/EXP-08_TCN_INCEPTION/METRICS.json` | PyTorch Weights: `models/multiscale_tcn_model.pt` |
| **EXP-09 (Isotonic Risk Calibration)**| Reliability & ECE Benchmark JSON | `experiments/EXP-09_CALIBRATION/METRICS.json` | Serialized Calibrator: `models/isotonic_calibrator.pkl` |
| **EXP-10 (Context-Aware Intelligence)**| Context Feature Ablation JSON | `experiments/EXP-10_CONTEXT_AWARENESS/METRICS.json` | `experiments/EXP-10_CONTEXT_AWARENESS/RESULTS.md` |
| **B3-DualStream (Independent EXP-04)**| Candidate Benchmark JSON | `results/teammate_candidate_exp04_results.json` | Evaluated via `experiments/run_candidate_exp04_fast.py` |
| **B3-DualStream (Canonical Holdout)**| Candidate Holdout Evaluation Script | `docs/POST_MERGE_INTEGRATION_REPORT.md` | Evaluated via `experiments/evaluate_teammate_candidate.py` |
| **B3 Metadata Contract** | Candidate Model Metadata | `models/candidates/candidate_metadata.json` | Candidate Models: `models/candidates/model_*.pkl` |

---

## 6. Provenance Notes & Discrepancy Reconciliation

During forensic audits of repository documentation, seven distinct discrepancies were identified across historical working drafts, markdown text snippets, and serialized benchmark ledgers. In accordance with strict release discipline, each discrepancy is traced back to its benchmark-producing source and formally reconciled below:

### Discrepancy 1: B2 EXP-04 Escalated Dies (2,027 vs 2,028)
* **Observed Conflict:** Certain narrative paragraphs reported 2,027 escalated dies, while `models/model_release.json` and `results/runtime_evaluation_latest.json` report 2,028 escalated dies.
* **Root Cause Analysis:** On the 32,598-die independent test set, the exact count of dies satisfying $0.295 \le P_A \le 0.550$ is **2,028 dies** ($2028 / 32598 = 6.221245\%$). In an early summary document, an author multiplied the rounded rate ($32,598 \times 0.0622 = 2,027.5956$) and truncated without ceiling rounding, producing 2,027.
* **Authoritative Resolution:** **2,028 dies** is the exact integer count verified by deterministic vector filtering in both `models/model_release.json` (line 56) and `experiments/EXP-04/independent_metrics.json`.

### Discrepancy 2: B2 EXP-04 Compute-Bypassed Count (30,571 vs 30,570)
* **Observed Conflict:** Early documentation referenced 30,571 fast-path dies, while `models/model_release.json` explicitly states 30,570 dies.
* **Root Cause Analysis:** This was the direct arithmetic corollary to Discrepancy 1 ($32,598 - 2,027 = 30,571$ vs $32,598 - 2,028 = 30,570$).
* **Authoritative Resolution:** **30,570 dies** ($93.7788\%$) is the mathematically exact count of fast-path dies ($30,102 \text{ fast pass} + 468 \text{ fast fail} = 30,570$).

### Discrepancy 3: B2 PR-AUC on Canonical Holdout (0.4777 vs 0.4779)
* **Observed Conflict:** `models/model_release.json` records $0.4777$, whereas `COMPLETE_METRICS.csv` recorded $0.4779$.
* **Root Cause Analysis:** `COMPLETE_METRICS.csv` computed PR-AUC using trapezoidal Riemann integration (`np.trapz` over sorted precision-recall points), whereas `model_release.json` and `AUTHORITATIVE_CLEAN_METRICS.json` use Scikit-Learn's standard interpolation formula `average_precision_score()`, which evaluates to $0.47768$ (rounding to $0.4777$).
* **Authoritative Resolution:** **0.4777** is the authoritative production standard mandated in `model_release.json` (line 94).

### Discrepancy 4: B2 Escalation Percentage on Canonical Holdout (5.97% vs 5.98%)
* **Observed Conflict:** EXP-02 reported an escalation rate of $5.97\%$, whereas B2 production in `model_release.json` reports $5.98\%$.
* **Root Cause Analysis:** In EXP-02, the preliminary candidate Model A gated exactly 1,670 dies ($1,670 / 27,972 = 5.97025\%$). In EXP-03 / EXP-03B, with the final frozen Model B weights, exactly 1,673 dies were escalated ($1,673 / 27,972 = 5.98098\%$). Truncating to two decimals gives 5.97%, while rounding gives 5.98%.
* **Authoritative Resolution:** **5.98%** (1,673 dies out of 27,972) is the true operating rate for production B2 as registered in `models/model_release.json` (line 102).

### Discrepancy 5: B3 Feature Count (517/528 vs older 518/529 references)
* **Observed Conflict:** Early integration drafts cited Model A having 518 features and Model B having 529 features, while the verified codebase report states 517 and 528.
* **Root Cause Analysis:** Physical artifact inspection of serialized models `models/candidates/model_A_enhanced.pkl` and `models/candidates/model_B_tab_enhanced.pkl` revealed `model_A.num_feature() == 517` and `model_B_tab.num_feature() == 528`. The older 518/529 numbers were from an unmerged branch that temporarily included `param_cnt_gt_4sigma` before feature redundancy pruning. The PyTorch 1D-CNN operates on the 2,000-block time series as a separate sequence modality and is not part of the tabular column array.
* **Authoritative Resolution:** **Model A = 517 features**, **Model B tabular = 528 features**, **CNN = separate 2,000-block modality**.

### Discrepancy 6: Candidate High-Recall Policy False Negatives (625 vs 845)
* **Observed Conflict:** Line 161 of `docs/POST_MERGE_INTEGRATION_REPORT.md` stated FN = 625 for the High-Recall policy, while `results/teammate_candidate_exp04_results.json` reports FN = 845.
* **Root Cause Analysis:** On `input/test.csv`, the total actual failure count is strictly 1,380 dies. With TP = 535, the true false negative count is $1,380 - 535 = \mathbf{845}$. The 625 figure was an editorial typo in the markdown documentation.
* **Authoritative Resolution:** **FN = 845** from `results/teammate_candidate_exp04_results.json` is the authoritative value.

### Discrepancy 7: Candidate Canonical Holdout False Positives (92 vs Screenshot 106)
* **Observed Conflict:** An exploratory screenshot referenced 106 False Positives for the teammate candidate on the canonical holdout, whereas the benchmark reproduction reported 92 False Positives.
* **Root Cause Analysis:** The screenshot captured an uncalibrated precursor before fixing threshold $\tau_B = 0.5200$. When executed deterministically using `experiments/evaluate_teammate_candidate.py`, the reproduced result is strictly **FP = 92** (yielding Precision = 81.78%).
* **Authoritative Resolution:** **FP = 92** is the verified benchmark-producing result.
