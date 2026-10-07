# EXP-08: Multi-Scale 1D Inception / Dilated TCN Architecture

## Objective
Investigate replacing the baseline 3-layer 1D CNN with a Multi-Scale Dilated Temporal Convolutional Network (TCN) featuring:
- Parallel dilated convolutions ($d=1, 2, 4$)
- Multi-scale temporal feature fusion
- Residual skip connections
- Parameter count: 39,329 (vs CNN: 41,153)

## Evaluated Modes
- **Mode A (Forensic Attribution)**: Standalone 2,000-block sequence classification and attribution.
- **Mode B (Late Fusion)**: Late probability-level blending with tabular Model B2.

## Summary Results
- **Mode A (Sequence Alone)**:
  - CNN: PR-AUC = 0.0794, ROC-AUC = 0.6698, F1 = 0.1206
  - TCN: PR-AUC = 0.0874, ROC-AUC = 0.6637, F1 = 0.1474
- **Mode B (Cascade Fusion)**:
  - Frozen B2 Baseline: Cascade F1 = 0.5264, PR-AUC = 0.4779
  - B2 + TCN Fusion: Cascade F1 = 0.5229, PR-AUC = 0.4794

## Scientific Decision
- Sequence Encoder: **TCN_SUPERIOR_SEQUENCE_ENCODER** (TCN demonstrates higher sequence PR-AUC).
- Cascade Production: **REJECTED_MODE_B_RETAIN_FROZEN_B2** (Retain B2 for decision making; TCN recommended for forensic attribution only).
