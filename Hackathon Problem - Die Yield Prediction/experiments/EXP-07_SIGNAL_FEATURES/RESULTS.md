# EXP-07 Experimental Results: Advanced Signal Features

## 1. Executive Summary
- **Hypothesis**: The screenshot suggested that Kurtosis, Run-Length, and FFT features could lift defect recall by capturing subtle anomaly signatures.
- **Empirical Findings**:
  1. **Ablation Comparison**:
     - Adding **Statistical Moments** alone (542 features) yielded Cascade F1 = **0.5190** and PR-AUC = **0.4732** (Delta F1 = **-0.0074**).
     - Adding **Run-Length Clusters** alone (566 features) yielded Cascade F1 = **0.5153** and PR-AUC = **0.4755** (Delta F1 = **-0.0110**).
     - Adding **FFT Spectral Descriptors** alone (544 features) yielded Cascade F1 = **0.5203** and PR-AUC = **0.4760** (Delta F1 = **-0.0061**).
     - Combining **All Families** (580 features) yielded Cascade F1 = **0.5180** and PR-AUC = **0.4726** (Delta F1 = **-0.0084**).
  2. **Feature Importance Forensics**:
     - The existing 10 candidate block features frozen in B2 (`max_roll20`, `cnt_110`, `cluster_area_110`) already dominate feature gain. The new FFT and Kurtosis descriptors provide redundant signal that LightGBM largely downweights.
- **Decision**: **REJECTED_KEEP_FROZEN_B2**. The proposed signal feature extensions do not demonstrate a meaningful performance lift over frozen B2.

## 2. Full Ablation Table

| Feature Set | Feature Count | Standalone F1 | Standalone PR-AUC | Cascade F1 | Cascade PR-AUC | Escalation % | Net TP Gain | Latency (ms/die) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B2 Baseline** | 536 | 0.4892 | 0.4771 | **0.5264** | **0.4763** | 5.97% | **45** | 0.0085 |
| **B2 + Statistical** | 542 | 0.4638 | 0.4626 | 0.5190 | 0.4732 | 5.97% | 41 | 0.0091 |
| **B2 + Run-Length** | 566 | 0.4531 | 0.4624 | 0.5153 | 0.4755 | 5.97% | 37 | 0.0090 |
| **B2 + FFT** | 544 | 0.4839 | 0.4783 | 0.5203 | 0.4760 | 5.97% | 42 | 0.0090 |
| **B2 + All Families** | 580 | 0.4587 | 0.4613 | 0.5180 | 0.4726 | 5.97% | 40 | 0.0089 |
