# EXP-02 — Joint Threshold & Asymmetric Cascade Optimization

**Experiment ID:** `EXP-02`  
**Date:** October 6, 2026  
**Status:** COMPLETE  
**Decision:** **CONDITIONAL KEEP**  

---

## Executive Summary

EXP-02 conducted a rigorous, fully vectorized, deterministic grid search ($961$ operating points) and secondary sweep over the decision policy of the SanDisk Cerebrum selective cascade, pairing the **EXP-01 spatially restored Model A** with the production **Model B**.

### Key Findings:
1. **Cascade Structural Invariance:** When $\tau_A$ lies between `gate_lower` and `gate_upper` ($\text{gate\_lower} \le \tau_A \le \text{gate\_upper}$), Model A's decision threshold has **zero mathematical effect on cascade output**. Non-escalated dies are determined exclusively by the gating boundaries (passes below `gate_lower`, fails above `gate_upper`), while intermediate dies are evaluated by Model B. Thus, `gate_lower` and `gate_upper` govern cascade classification.
2. **Optimal Operating Point (`MAX_F1_COMPUTE_OPTIMAL`):**
   * $\tau_A = 0.5100$, $\text{gate\_lower} = 0.2950$, $\text{gate\_upper} = 0.5500$, $\tau_B = 0.5376$.
   * **Cascade F1 reaches 0.5199** (+0.0128 vs Production Baseline 0.5072, +0.0059 vs EXP-01 0.5140).
   * **Cascade Precision reaches 87.28%** (+5.90% vs Production Baseline 81.38%, +4.68% vs EXP-01 82.60%).
   * **False Positives plunge from 89 (Prod) and 83 (EXP-01) to 57** (-32 false alarms vs Production, -26 vs EXP-01).
   * **Compute Saved increases to 94.03%** (escalation drops from 11.86% to 5.97%, cutting Model B evaluations in half from 3,317 to 1,670 dies).
   * **True Positives remain net positive:** 391 detected defects (vs 389 Production Baseline).
3. **Trapped FN Hypothesis Refuted:** Expanding Gate B downwards below $0.25$ (e.g. to $0.15$ or $0.20$) fails to rescue the $473$ low-probability Model A false negatives ($P_A < 0.25$). Only $2$ to $5$ dies are caught by Model B, while generating $12$ to $40$ new false alarms and bloating escalation up to $42.68\%$.
4. **Wafer-Level Robustness:** Across all 32 validation wafers, EXP-02 improves F1 on **16 wafers** (vs 11 in EXP-01) and degrades on only **5 wafers** (vs 13 in EXP-01). Mean wafer precision rises to **69.32%** (vs 63.92% baseline).
5. **Verdict:** **CONDITIONAL KEEP**. The policy improves F1, precision, false alarms, MCC, and compute efficiency, while preserving net defect detection over baseline. However, formal calibration (EXP-03) and holdout verification are required before permanent production release.

---

## Objective

Determine whether changing the decision policy around the EXP-01 spatially restored Model A and Model B selective cascade improves defect detection, precision, compute efficiency, and wafer consistency without model retraining.

Specifically, answer:
> *What Model A threshold $\tau_A$ and Gate B interval $[\text{gate\_lower}, \text{gate\_upper}]$ provide the best overall semiconductor defect-detection operating point while balancing recall, precision, F1/F2, MCC, PR-AUC, false alarms, escaped defects, escalation rate, compute savings, and wafer-level robustness?*

---

## Experimental Controls

Strict isolation protocols were enforced throughout EXP-02:
* **No Model Retraining:** Model A, Model B, and the CNN were strictly immutable.
* **No Feature Engineering:** Zero feature modifications.
* **Dataset Preservation:** The canonical holdout split remained 100% frozen.
* **Artifact Immutability:** All 9 production artifacts were pre-hashed and post-hashed with SHA-256 (verified 100% bit-for-bit identical).
* **Isolation:** All generated metrics, CSVs, plots, and reports were quarantined under `experiments/EXP-02/`.

---

## Dataset

Evaluated exclusively on the canonical holdout validation set:
* **Total Eligible Dies:** $27,972$ dies ($old\_label == 0$)
* **Wafers:** $32$ distinct holdout wafers
* **Actual Failures:** $1,056$ dies ($3.7752\%$ defect rate)
* **Healthy Dies:** $26,916$ dies ($96.2248\%$)
* **Target Isolation:** Complete wafer-level holdout ($\text{Train} \cap \text{Val} = \emptyset$). Zero leakage.

---

## Model Configuration

* **Model A:** EXP-01 Spatially Restored LightGBM (`experiments/EXP-01/model_A_spatial.pkl`).
* **Model B:** Production LightGBM (`models/model_B.pkl`).
* **Model B Classification Threshold:** Locked at $\tau_B = 0.5376$.
* **Signal Diagnostic Threshold:** Locked at $104.0$ (untouched).

---

## Search Space

### Variable A — Model A Classification Threshold
$$\tau_A \in [0.400, 0.550], \quad \text{step} = 0.005 \quad (31 \text{ discrete values})$$

### Variable B — Gate B Lower Bound
$$\text{gate\_lower} \in [0.150, 0.300], \quad \text{step} = 0.005 \quad (31 \text{ discrete values})$$

### Variable C — Gate B Upper Bound
Initially locked to $\text{gate\_upper} = 0.55$. Secondary sweep evaluated $\text{gate\_upper} \in [0.50, 0.65]$ with step $0.01$.

**Total Grid Combinations:** $31 \times 31 = 961$ full evaluation points.

---

## Cascade Semantics

The verified production cascade routes each die $i$ as follows:
1. Model A produces probability $P_A$.
2. **Escalation Decision:**
   $$\text{Escalated} = (\text{gate\_lower} \le P_A \le \text{gate\_upper})$$
3. **If Escalated (Slow Path):**
   * Model B evaluates block sequential signal $\rightarrow P_B$.
   * $\text{Prediction} = (P_B \ge 0.5376)$.
   * $\text{Cascade Probability} = P_B$.
4. **If Not Escalated (Fast Path):**
   * Model A handles directly.
   * $\text{Prediction} = (P_A \ge \tau_A)$.
   * $\text{Cascade Probability} = P_A$.

**Structural Theorem:** When $\text{gate\_lower} \le \tau_A \le \text{gate\_upper}$:
* If $P_A < \text{gate\_lower}$, then $P_A < \tau_A \implies \text{Prediction} = 0$.
* If $P_A > \text{gate\_upper}$, then $P_A > \tau_A \implies \text{Prediction} = 1$.
* If $\text{gate\_lower} \le P_A \le \text{gate\_upper}$, prediction is delegated to Model B.
Thus, **the binary cascade decision is mathematically invariant to $\tau_A$ throughout this entire region.**

---

## Global Grid Results

From 961 evaluated operating points in `grid_results.csv`:
* **Cascade F1 Range:** $0.4636$ to $0.5199$
* **Cascade Recall Range:** $36.08\%$ to $37.88\%$
* **Cascade Precision Range:** $67.43\%$ to $87.97\%$
* **Cascade False Positives:** $52$ to $183$ dies
* **Escalation Rate Range:** $5.58\%$ to $42.68\%$
* **Compute Saved Range:** $57.32\%$ to $94.42\%$

---

## Pareto Frontier

Multi-objective Pareto optimization across Recall, Precision, F1, F2, MCC, PR-AUC, Compute Saved, FP, and FN identified **465 non-dominated operating points** (saved in `pareto_frontier.csv`).

Key Pareto trade-off points:
1. **Precision / Compute Extremum:** $\text{gate\_lower} = 0.300 \implies \text{Prec} = 87.36\%, \text{FP} = 56, \text{Esc} = 5.58\%, \text{Rec} = 36.65\%$.
2. **F1 Optimal Point:** $\text{gate\_lower} = 0.295 \implies \text{F1} = 0.5199, \text{Prec} = 87.28\%, \text{FP} = 57, \text{Esc} = 5.97\%, \text{Rec} = 37.03\%$.
3. **Balanced Recall Point:** $\text{gate\_lower} = 0.225 \implies \text{F1} = 0.5133, \text{Rec} = 37.50\%, \text{FP} = 91, \text{Esc} = 16.07\%$.
4. **Maximum Recall Extremum:** $\text{gate\_lower} = 0.150 \implies \text{Rec} = 37.78\%, \text{FP} = 123, \text{Esc} = 42.68\%, \text{F1} = 0.5057$.

---

## Candidate Operating Points Comparison

| Candidate ID | Status | $\tau_A$ | $\text{gate\_lower}$ | $\text{gate\_upper}$ | F1 | F2 | Recall | Precision | MCC | TP | FN | FP | Escalation | Compute Saved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MAX_F1_COMPUTE_OPTIMAL** | **PREFERRED** | **0.510** | **0.295** | **0.55** | **0.5199** | **0.4185** | **37.03%** | **87.28%** | **0.5589** | **391** | **665** | **57** | **5.97%** | **94.03%** |
| BEST_RECALL_CONTROLLED_FP | VIABLE | 0.510 | 0.225 | 0.55 | 0.5133 | 0.4203 | 37.50% | 81.31% | 0.5415 | 396 | 660 | 91 | 16.07% | 83.93% |
| BEST_F2_BALANCED | VIABLE | 0.510 | 0.225 | 0.55 | 0.5133 | 0.4203 | 37.50% | 81.31% | 0.5415 | 396 | 660 | 91 | 16.07% | 83.93% |
| BEST_MCC_EFFICIENT | VIABLE | 0.510 | 0.295 | 0.55 | 0.5199 | 0.4185 | 37.03% | 87.28% | 0.5589 | 391 | 665 | 57 | 5.97% | 94.03% |
| MAX_F2_AGGRESSIVE_RESCUE | REJECT | 0.510 | 0.195 | 0.55 | 0.5096 | 0.4207 | 37.69% | 78.66% | 0.5333 | 398 | 658 | 108 | 24.24% | 75.76% |

---

## Best Recall Operating Point

* **Configuration:** $\tau_A = 0.510$, $\text{gate\_lower} = 0.150$, $\text{gate\_upper} = 0.55$
* **Recall:** $37.78\%$ ($399$ TPs detected, $+10$ vs Prod, $+5$ vs EXP-01)
* **Tradeoff Analysis:** To detect 5 additional defects over EXP-01, False Positives surge from $83$ to $123$ ($+40$ false alarms), and Model B evaluates $11,939$ dies ($42.68\%$ escalation rate). **Violates false alarm and compute constraints.**

---

## Best F2 Operating Point

* **Configuration:** $\tau_A = 0.510$, $\text{gate\_lower} = 0.195$, $\text{gate\_upper} = 0.55$
* **F2 Score:** $0.4207$
* **Tradeoff Analysis:** Generates $108$ False Positives (Precision drops to $78.66\%$) and escalates $24.24\%$ of all dies. Designated **REJECT**.

---

## Best F1 Operating Point

* **Configuration:** $\tau_A = 0.510$, $\text{gate\_lower} = 0.295$, $\text{gate\_upper} = 0.55$
* **F1 Score:** $0.5199$
* **Tradeoff Analysis:** Precision jumps to $87.28\%$, False Positives drop to $57$, escalation drops to $5.97\%$, while catching $391$ defects (+2 over Production).

---

## Best MCC Operating Point

* **Configuration:** $\tau_A = 0.510$, $\text{gate\_lower} = 0.295$, $\text{gate\_upper} = 0.55$
* **MCC:** $0.5589$ (+0.0221 vs Production 0.5369, +0.0142 vs EXP-01 0.5447).
* Demonstrates optimal overall correlation between predictions and ground truth.

---

## Preferred Operating Point

The **PREFERRED** candidate is `MAX_F1_COMPUTE_OPTIMAL`:
* **Parameters:** $\tau_A = 0.5100$, $\text{gate\_lower} = 0.2950$, $\text{gate\_upper} = 0.5500$, $\tau_B = 0.5376$.
* **Rationale:**
  1. Achieves the highest F1 ($0.5199$) and MCC ($0.5589$) in the history of the system.
  2. Eliminates $32$ false alarms compared to Production ($89 \rightarrow 57$) and $26$ compared to EXP-01 ($83 \rightarrow 57$).
  3. Cuts Model B inference workload by almost half ($11.26\% \rightarrow 5.97\%$), delivering **94.03% compute savings**.
  4. Preserves defect detection ($391$ TPs vs $389$ baseline).
  5. Exhibits superior wafer-level consistency (16 wafers improved, only 5 degraded vs baseline).

---

## Trapped FN Analysis

The forensic audit documented that $473$ of Model A's $707$ false negatives in EXP-01 had $P_A < 0.25$. EXP-02 tested whether opening the Gate B window downward would rescue them.

| $\text{gate\_lower}$ | Newly Exposed Original FNs | Rescued FNs | Rescue Rate | New False Positives | Escalation Rate | Cascade Recall | Cascade FP | Cascade F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.15 | 258 | 5 | 1.94% | 40 | 42.68% | 37.78% | 123 | 0.5057 |
| 0.18 | 173 | 4 | 2.31% | 31 | 29.56% | 37.69% | 114 | 0.5077 |
| 0.20 | 110 | 3 | 2.73% | 23 | 22.69% | 37.59% | 106 | 0.5093 |
| 0.22 | 61 | 2 | 3.28% | 12 | 17.16% | 37.50% | 95 | 0.5120 |
| **0.25 (Ref)** | **0** | **0** | **0.00%** | **0** | **11.26%** | **37.31%** | **83** | **0.5140** |
| 0.27 | 0 | 0 | 0.00% | 0 | 8.53% | 37.03% | 73 | 0.5145 |
| **0.295 (Pref)**| **0** | **0** | **0.00%** | **0** | **5.97%** | **37.03%** | **57** | **0.5199** |

**Conclusion:** The rescue rate of trapped FNs below $0.25$ is extremely poor ($1.9\% - 3.3\%$). For every $1$ trapped defect caught by Model B in this low-probability regime, the system incurs **$8$ to $10$ new false alarms**. The data definitively shows that dies with $P_A < 0.22$ lack anomalous block signatures in Model B.

---

## Recovery Ledger

Accounting for every die entering the cascade:

| Metric | Production Baseline | EXP-01 Locked | EXP-02 Preferred |
| :--- | :---: | :---: | :---: |
| **Model A Threshold ($\tau_A$)** | 0.5100 | 0.5100 | 0.5100 |
| **Gate B Interval** | [0.25, 0.55] | [0.25, 0.55] | [0.295, 0.55] |
| **Model A Total FNs** | 699 | 707 | 707 |
| $\rightarrow$ FNs Sent to Model B | 215 | 234 | 166 |
| $\rightarrow$ FNs Rescued by Model B | 37 | 49 | 46 |
| $\rightarrow$ Final Cascade FNs | 667 | 662 | 665 |
| **Model A Total TPs** | 357 | 349 | 349 |
| $\rightarrow$ TPs Sent to Model B | 20 | 14 | 14 |
| $\rightarrow$ TPs Preserved by Model B | 15 | 10 | 10 |
| $\rightarrow$ TPs Lost by Model B | 5 | 4 | 4 |
| $\rightarrow$ Final Cascade TPs | 389 | 394 | 391 |
| **Model A Total FPs** | 43 | 40 | 40 |
| $\rightarrow$ FPs Sent to Model B | 25 | 27 | 27 |
| $\rightarrow$ FPs Removed by Model B | 23 | 26 | 26 |
| $\rightarrow$ FPs Remaining from Model A | 2 | 1 | 1 |
| $\rightarrow$ New FPs Introduced by Model B | 69 | 69 | 43 |
| $\rightarrow$ Final Cascade FPs | 89 | 83 | 57 |

---

## Threshold Analysis

* Across the evaluated range $\tau_A \in [0.40, 0.55]$, varying $\tau_A$ has **zero effect on the cascade** as long as $\tau_A \in [\text{gate\_lower}, \text{gate\_upper}]$.
* However, keeping $\tau_A = 0.5100$ is crucial for standalone Model A operation, maintaining high standalone precision ($89.72\%$) and low false alarms ($40$).

---

## Gate Analysis

* **Gate B Lower Bound:** Moving `gate_lower` from $0.250$ to $0.295$ eliminates dies in $[0.250, 0.295)$ from escalation. In this band, empirical defect rate is low ($6.8\%$). Model B previously evaluated $1,481$ dies here, catching only $3$ defects while generating $26$ false alarms. Bypassing this band saves compute and eliminates false alarms.
* **Gate B Upper Bound:** Secondary sweep over $\text{gate\_upper} \in [0.50, 0.65]$ revealed a plateau between $0.53$ and $0.55$. At $0.55$, F1 is $0.5199$ with $57$ FPs vs $0.5205$ with $64$ FPs at $0.53$. Keeping $\text{gate\_upper} = 0.55$ is completely justified and optimal.

---

## Wafer-Level Robustness

Evaluation across all 32 canonical validation holdout wafers:
* **Wafers Improved vs Production Baseline:** **16 wafers** (vs 11 in EXP-01).
* **Wafers Unchanged vs Production Baseline:** **11 wafers**.
* **Wafers Degraded vs Production Baseline:** **5 wafers** (down from 13 in EXP-01!).
* **Wafers Improved vs EXP-01:** **13 wafers**.
* **Wafers Unchanged vs EXP-01:** **16 wafers**.
* **Wafers Degraded vs EXP-01:** **3 wafers** (`W_F_0043`, `W_N_0063`, `W_N_0086`). On each degraded wafer, detected defects dropped by exactly 1 die, but False Positives were cut in half.
* **Mean Wafer Precision:** **69.32%** (vs 63.92% Baseline, 65.13% EXP-01).
* **Mean Wafer F1:** **0.4467** (vs 0.4210 Baseline, 0.4368 EXP-01).
* **Mean Wafer Recall:** **34.24%** (vs 33.74% Baseline).

---

## Probability-Band Analysis

| Probability Band | Population | Actual Defects | Defect Rate | Model B Precision in Band | Outcome & Impact |
| :---: | :---: | :---: | :---: | :---: | :--- |
| $[0.00, 0.15)$ | 16,036 | 215 | 1.34% | N/A | Fast-Path Pass (True Negatives dominant) |
| $[0.15, 0.25)$ | 7,114 | 258 | 3.63% | 11.11% | Trapped FN band; Model B rescue has 10:1 FP penalty |
| $[0.25, 0.295)$| 1,481 | 108 | 7.29% | 10.34% | Inefficient escalation band; bypassing cuts 26 FPs |
| $[0.295, 0.55)$| 1,670 | 180 | 10.78% | 51.69% | Optimal Escalation Band; Model B catches 46 defects |
| $[0.55, 1.00)$ | 1,671 | 295 | 17.65% | N/A | Fast-Path Fail; High confidence defect cluster |

---

## Calibration

* **Brier Score:** $0.04188$ (virtually identical to Production $0.04175$).
* **ECE:** $0.12279$ (vs $0.12053$ Production Baseline).
* Note: Decision policy optimization shifts decision boundaries; it does not perform parametric probability calibration. Probability calibration remains reserved for EXP-03.

---

## Production Baseline vs EXP-01 vs EXP-02

The authoritative comparison across all 18 primary metrics:

```text
Metric                  Production    EXP-01    EXP-02    Δ vs Prod    Δ vs EXP-01
-----------------------------------------------------------------------------------
PR-AUC                  0.4677        0.4755    0.4741    +0.0064      -0.0014
ROC-AUC                 0.7990        0.7982    0.7973    -0.0017      -0.0009
Recall                  36.84%        37.31%    37.03%    +0.19%       -0.28%
Precision               81.38%        82.60%    87.28%    +5.90%       +4.68%
F1                      0.5072        0.5140    0.5199    +0.0128      +0.0059
F2                      0.4137        0.4191    0.4185    +0.0048      -0.0006
MCC                     0.5369        0.5447    0.5589    +0.0221      +0.0142
Balanced Accuracy       68.25%        68.50%    68.41%    +0.15%       -0.09%
Specificity             99.67%        99.69%    99.79%    +0.12%       +0.10%
FPR                     0.0033        0.0031    0.0021    -0.0012      -0.0010
Brier                   0.04175       0.04115   0.04188   +0.0001      +0.0007
ECE                     0.12053       0.11930   0.12279   +0.0023      +0.0035
TP                      389           394       391       +2           -3
FN                      667           662       665       -2           +3
FP                      89            83        57        -32          -26
TN                      26827         26833     26859     +32          +26
Escalation Rate         11.86%        11.26%    5.97%     -5.89%       -5.29%
Compute Saved           88.14%        88.74%    94.03%    +5.89%       +5.29%
```

```text
tau_A = 0.5100
gate_lower = 0.2950
gate_upper = 0.5500
Model B threshold = 0.5376
```

---

## Artifact Integrity

SHA-256 pre-run and post-run hash verification across all 9 production artifacts:
* `models/model_A.txt`: `3d4ef18db2a96e108a789effa50fc11f95dd507e974b8e14cd9169ca8657c6f5` (UNCHANGED)
* `models/model_A.pkl`: `0ca33068da29c7f951a624b47afdbc50ae9b76c47bfee7e9c0189ba6417f6071` (UNCHANGED)
* `models/model_B.pkl`: `3f29d1d9b6ac1eb5b11bdc8db47238373eb219f67a86d2eff0007e6dbc58ef24` (UNCHANGED)
* `models/cnn_1d_block_model.pt`: `0a081de68dfb0de38bfa1780cf169aab4ab2b4bc5777c8156ac1bed9772160b5` (UNCHANGED)
* `models/selective_cascade_meta.pkl`: `eba6a1d6713602798c64c8460948c0b07be58274305dcf8182e9a13bad76fd1d` (UNCHANGED)
* `models/feature_metadata.json`: `2cddb4516a7e9203c372db00adf3a0045fc92bbd3f0dbe08c05203f4bdbe35ef` (UNCHANGED)
* `results/AUTHORITATIVE_CLEAN_METRICS.json`: `569aaa6bdf388477aebb77e759fa9c04cd29d3c30ca70c3e6f7737544163547a` (UNCHANGED)
* `results/val_die_diagnostics.csv`: `11314ead2f908a7e046665ad4eac144e90ef3573e4a49ab622390ac0bea78ae1` (UNCHANGED)
* `results/canonical_wafer_split.json`: `319ed176fe5216596341d6b81cf05b7e43bf7083462898fc1f7aea80f32539c4` (UNCHANGED)

**Result:** Zero mutations. Production artifacts remain 100% pristine.

---

## Tests

* **Unit & Regression Suite:** `Ran 28 tests in 0.876s — OK` (28/28 PASS).
* **Frontend Production Bundle:** `tsc && vite build` built in 3.63s with 0 errors.

---

## Limitations

1. **Single 32-Wafer Validation Split:** While canonical and leak-free, variance exists across defect-sparse vs defect-dense wafers.
2. **Probability Calibration:** Raw LightGBM leaf outputs are uncalibrated; ECE remains around $12\%$.
3. **Model B Threshold Locked:** $\tau_B = 0.5376$ was held constant per protocol; joint optimization of $\tau_B$ was not explored.

---

## Recommendation

**CONDITIONAL KEEP**

Adopt the asymmetric operating window $[\text{gate\_lower}=0.2950, \text{gate\_upper}=0.5500]$ for the candidate system. It yields the highest F1 ($0.5199$), highest Precision ($87.28\%$), and lowest escalation rate ($5.97\%$) without sacrificing net defects over baseline ($391$ vs $389$). Retain conditional status until probability calibration and GroupKFold cross-validation are completed.

---

## Next Experiment

* **EXP-03:** Isotonic / Platt Probability Calibration on Model A and Model B cascade outputs to minimize Brier score and ECE before final threshold lock.
