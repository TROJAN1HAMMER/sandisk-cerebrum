# SanDisk Die Yield Prediction — Selective Multi-Resolution System

An interpretable multi-resolution die-yield prediction system that combines broad electrical parametric/spatial evidence with targeted fine-resolution sub-die block inspection.

---

## 1. What the Project Does
Predicts post-test die failures on semiconductor wafers before final assembly and packaging. It filters pre-test passing dies (`old_label == 0`) and predicts newly failing dies (`label == 1`), optimizing yield and preventing costly packaging of defective dies.

## 2. Why the Problem is Difficult
* **Parametric Shift Suppression**: 65% of post-test failures are marginal with suppressed electrical parametric shifts, causing severe overlap with passing dies in 500-dimensional feature space.
* **Sub-Die Block Sequence Noise**: Sequential 2,000 sub-die block readings contain un-masked physical anomaly signals, but are noisy ($N(0, 1.0)$ with non-failing spikes up to 104+). Indiscriminate global evaluation pollutes precision with false alarms.

## 3. System Architecture
```text
500 Parametric Measurements + Spatial Context
                     ↓
                  MODEL A
             Coarse Resolution
                     ↓
          Risk Probability P(Model A)
                     ↓
┌────────────────────┴────────────────────┐
│                                         │
Confident Region                          Uncertain Region (Gate C)
(P_A < 0.30 or P_A > 0.60)               (0.30 <= P_A <= 0.60)
│                                         │
▼                                         ▼
Final Decision                        2,000 Sub-Die Blocks
(Trust Model A)                                  ↓
                                              1D ConvNet
                                                 ↓
                                      Fine Resolution Evidence
                                                 ↓
                                      Calibrated Meta-Classifier
                                                 ↓
                                        Refined Decision
```

## 4. Model A Baseline (Authoritative Clean Benchmark)
LightGBM GBDT operating on 500 electrical parameters + 6 spatial features (`radial_dist`, `density_3x3`, `density_5x5`, `edge_dist`, `die_row`, `die_col`). Evaluated on 32 canonical out-of-fold validation wafers (27,972 eligible dies, zero train overlap):
* **AUC-PR**: **0.4493**
* **Fail F1**: **0.4900**
* **Precision**: **89.03%**
* **Recall**: **33.81%**
* **ROC-AUC**: **0.7927**
* **Confusion Matrix**: TP=357, FN=699, FP=44, TN=26,872 (at locked threshold 0.51)

## 5. Fine-Resolution Model B
A 3-layer 1D ConvNet (`Block1DCNN`) operating on raw $2,000 \times 1$ sub-die block sequences with a receptive field $\ge 100$ blocks, extracting 32-dimensional anomaly embeddings to address Model A's 699 missed false negatives.

## 6. Selective Multi-Resolution Cascade
Targets the 1D ConvNet block inspection strictly at Model A's uncertain dies (**$0.30 \le P_{\text{Model A}} \le 0.60$**). Dies outside this interval are classified directly by Model A. Escalated dies use a calibrated meta-classifier fusing $P_A$, $P_{\text{CNN}}$, and CNN block embeddings.

## 7. Performance & Benchmarks Status
> [!IMPORTANT]
> **Validation Hardening Notice**: An earlier evaluation quoting Model A F1 = 0.8789 / AUC-PR = 0.9448 was identified as **INVALID** due to train-validation wafer overlap (`results/canonical_wafer_split.json` vs RNG generator divergence).
> 
> The authoritative locked Model A baseline is **F1 = 0.4900 / AUC-PR = 0.4493**. Multi-resolution Model B and the Selective Cascade are being rebuilt and re-evaluated strictly on the canonical clean 128/32 split. See `results/MODEL_A_DISCREPANCY_AUDIT.md` and `results/QUARANTINED_LEAKED_RESULTS.md` for complete audit details.

## 8. Explainability
* **Coarse Level**: Feature importance ranking by LightGBM gain.
* **Fine Level**: 1D block occlusion sensitivity attribution identifying exact sub-die anomaly regions (e.g. sub-die blocks `~450–550`).
* **Cascade Decision Level**: Natural language escalation explanations in UI and CLI explaining why fine-resolution inspection was triggered or bypassed.

## 9. Installation & Setup
```bash
python3 -m pip install lightgbm shap torch pyarrow PyYAML pandas numpy scipy matplotlib streamlit
```

## 10. Training Pipeline Execution
To train Model A, Model B, and the Selective Cascade:
```bash
python3 build_model_A.py
python3 build_model_B.py
python3 build_selective_cascade.py
```

## 11. Production CLI Inference
To run fast batch inference on any input wafer dataset CSV:
```bash
python3 predict.py --input input/train.csv --output results/final_cli_test.csv
```

Outputs columns: `wafer_id`, `die_row`, `die_col`, `model_A_risk_prob`, `cascade_final_prob`, `model_A_decision`, `cascade_final_decision`, `resolution_used`, `escalation_reason`, `risk_category`.

## 12. Interactive Streamlit Web Dashboard
Launch the interactive 4-tab web platform:
```bash
streamlit run app.py
```
Tabs included:
1. **Executive Dashboard**: System KPI cards, yield metrics, and cascade architecture diagram.
2. **Wafer Spatial Risk Maps**: 4-panel side-by-side maps (Actual, Model A Risk, Final Cascade Risk, Final Classification).
3. **Die Inspector**: Interactive die picker, Model A vs Cascade risk breakdown, escalation explanation, and 1D block occlusion attribution plot.
4. **Model Benchmarks**: Full performance comparison tables and probability band analysis.
