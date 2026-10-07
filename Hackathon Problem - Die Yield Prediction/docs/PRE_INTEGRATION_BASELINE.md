# PRE-INTEGRATION BASELINE SNAPSHOT

**Platform:** SanDisk Cerebrum Die Yield Intelligence & Cascade Optimization Platform  
**Snapshot Date:** October 7, 2026 (Pre-Integration Frozen Baseline)  
**Purpose:** Cryptographic and operational safety record prior to evaluating and merging experimental candidate from `To_be_added/`.

---

## 1. Frozen Production Configuration

| Component | Production Release Value | Status |
| :--- | :--- | :--- |
| **Active Model Version** | `B2` (`2.0.0-b2`) | Frozen Production Candidate |
| **Active Model A** | LightGBM Classifier (506 features: 500 electrical + 6 spatial) | `models/model_A.pkl` |
| **Active Model B2** | LightGBM Classifier (536 features: 506 + 20 legacy + 10 candidate block) | `models/model_B.pkl` |
| **Model A Threshold ($\tau_A$)** | `0.5100` | Frozen |
| **Model B2 Threshold ($\tau_B$)**| `0.5376` | Frozen |
| **Active Production Gate Window**| `[0.295, 0.550]` | Locked (Emergency fallback: `0.295`) |
| **Historical Baseline Gate Window** | `[0.250, 0.550]` | Preserved for Phase 3C validation audits |
| **TCN Forensic Role** | Purely Offline Diagnostic / Saliency | `models/multiscale_tcn_model.pt` |
| **Calibrator Role** | Isotonic Probability Calibration for Reporting | `models/isotonic_calibrator.pkl` |

---

## 2. Cryptographic Checksums of Production Binaries

```bash
# Model Binaries SHA-256 Hashes
0ca33068da29c7f951a624b47afdbc50ae9b76c47bfee7e9c0189ba6417f6071  models/model_A.pkl
8877db7b55abcb8a7c28b9de6f662bb3dfacbbcfbdcacac6da0deeaf5314e09a  models/model_B.pkl
94e1fa2e65492cdb8ebb21b85a38390450f5a31bd1ef71e144c8850862dcf35a  models/multiscale_tcn_model.pt
0f8f2552a154374eb934b288640878c761b8db6be3357377df43abeb441cad7a  models/isotonic_calibrator.pkl
d2b76034528aa1c51131848c347d57bda0a3d7981a26db369cee311a119efc68  models/model_release.json
```

---

## 3. Verified Performance Metrics (EXP-04 Independent Test Set)

* **Population:** 40 unseen wafers, 32,598 dies, 1,380 defects ($4.2334\%$ prevalence)
* **Precision:** **$85.22\%$** ($513 / 602$)
* **Recall:** **$37.17\%$** ($513 / 1,380$)
* **Cascade F1 Score:** **$0.5177$** (95% Bootstrap CI: `[0.4948, 0.5369]`)
* **Cascade F2 Score:** **$0.4190$**
* **MCC:** **$0.5517$**
* **PR-AUC / ROC-AUC:** **$0.4808$ / $0.8078$**
* **Confusion Matrix:** $\text{TP}=513, \text{FN}=867, \text{FP}=89, \text{TN}=31,129$
* **Escalation Rate:** **$6.22\%$** (2,028 dies evaluated by Model B2)
* **Secondary Compute Saved:** **$93.78\%$** (30,570 dies fast-path bypassed)
* **FN Rescue Rate:** **$23.30\%$** ($41 / 176$ gated false negatives)
* **Net TP Recovery:** **$+37$** (41 Model-A FNs rescued, 4 Model-A TPs lost, net +37 TP)

---

## 4. Software & Platform Status

* **Backend Unit Tests:** **38 / 38 PASS** (`python3 -m unittest discover tests` in $0.83\text{s}$)
* **Frontend Compilation:** **0 TypeScript Errors, 0 Build Errors** (`npm run build` in $3.67\text{s}$)
* **Git Safety Branch:** `pre_integration_b2_baseline` created.
