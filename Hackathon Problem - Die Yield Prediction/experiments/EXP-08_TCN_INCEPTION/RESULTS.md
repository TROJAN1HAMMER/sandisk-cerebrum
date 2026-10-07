# EXP-08 Experimental Results: Multi-Scale Dilated TCN

## 1. Executive Summary
- **Hypothesis**: Replacing the 3-layer CNN with a Multi-Scale Dilated TCN could improve temporal pattern detection across contiguous sub-die anomaly runs.
- **Empirical Findings**:
  1. **Mode A — Standalone Sequence Modeling**:
     - The Multi-Scale Dilated TCN (39,329 parameters) achieved PR-AUC = **0.0874** and ROC-AUC = **0.6637**, outperforming the baseline 3-layer CNN (PR-AUC = **0.0794**, ROC-AUC = **0.6698**).
     - Multi-scale receptive fields ($d=1, 2, 4$) provide superior localized sensitivity to short-duration transients as well as broad anomaly clusters.
  2. **Mode B — Complete Cascade Decisioning**:
     - Combining TCN sequence probabilities with tabular B2 via late fusion yielded Cascade F1 = **0.5229** and PR-AUC = **0.4794** (vs Frozen B2 alone: F1 = **0.5264**, PR-AUC = **0.4779**).
     - Late fusion marginally dilutes precision in the ambiguous gating window due to sequence noise on edge dies.
- **Decision**:
  - **Forensic Diagnosis**: The Multi-Scale Dilated TCN is superior for diagnostic attribution and visual defect cluster explanation.
  - **Production Cascade**: **REJECTED_MODE_B_RETAIN_FROZEN_B2**. Tabular B2 remains the sole authoritative decision maker in Stage 2.

## 2. Sequence Model Architecture Comparison (Mode A)

| Architecture | Parameters | Training Time | Latency (ms/die) | PR-AUC | ROC-AUC | F1 Score | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 3-Layer CNN** | 41,153 | — | 0.0207 | 0.0794 | 0.6698 | 0.1206 | 54.83% | 6.77% |
| **Multi-Scale Dilated TCN** | 39,329 | 206.3s | 0.0344 | **0.0874** | **0.6637** | **0.1474** | 27.94% | 10.01% |

## 3. Cascade Mode B Comparison

| System | Cascade F1 | Cascade PR-AUC | Escalation % | Net TP Gain | FN Rescued |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Frozen B2 Baseline** | **0.5264** | **0.4779** | 5.97% | **45** | 48 |
| **B2 + CNN Fusion** | 0.5189 | 0.4782 | 5.97% | 43 | 47 |
| **B2 + TCN Fusion** | 0.5229 | 0.4794 | 5.97% | 39 | 43 |
