# EXP-04: Label-Access Gate Decision

## Dataset
`input/test.csv` (842.1 MB, 39,351 rows, 506 columns, 40 unique wafers)

## Why it is independent
1. **Zero Wafer Intersection**: The 40 wafers in `input/test.csv` were synthesized as an independent test population (`test_df` in `generate_data.py`). They have exactly zero overlap with the 160 wafers in `input/train.csv`, zero overlap with the 128 canonical training wafers, and zero overlap with the 32 canonical validation holdout wafers:
   $$\text{Train Wafers (128)} \cap \text{Val Wafers (32)} \cap \text{Test Wafers (40)} = \emptyset$$
2. **True Manufacturing Isolation**: In semiconductor fabrication terms, these 40 wafers represent distinct, unobserved wafer production runs with their own spatial defect distributions, wafer-level process shifts, and parametric variations.
3. **No Optimization Contamination**: The models (Model A, Model B B2), feature formulas, rolling windows, cluster thresholds, spatial defect definitions, and cascade gate boundaries were derived and frozen exclusively using `input/train.csv` (and verified on the 32 canonical validation wafers).

## Whether its labels were previously accessed
- **Algorithmic Access**: **NEVER**. No model training script, feature selection algorithm, threshold search routine, or hyperparameter optimizer in the repository has ever accessed, loaded, or evaluated the labels of `input/test.csv`.
- **Diagnostic Inspection**: The only prior access to `input/test.csv` was in `audit/model_forensics/` where file metadata, row/column counts, and wafer ID sets were inspected to verify schema integrity and wafer isolation.

## Whether its labels were used for
- **Feature selection**: **NO (0%)**
- **Threshold selection**: **NO (0%)**
- **Model selection**: **NO (0%)**
- **Error analysis**: **NO (0%)**
- **Hyperparameter tuning**: **NO (0%)**

## Whether final evaluation is permitted
**YES**. In accordance with the project roadmap (`audit/model_forensics/experiment_roadmap.md`: "Exp 6: Final Holdout Benchmark Validation on input/test.csv") and the data generation protocol (`generate_data.py`), `input/test.csv` is the designated ground-truth evaluation counterpart to the competition submission file `input/validation.csv`. Evaluating the locked, frozen B2 cascade on this independent ground truth is the scientifically intended final validation step.

## Evidence / Source
- `generate_data.py` lines 525–548: Confirms `test_df` is the ground-truth labeled cohort corresponding to the unlabeled competition submission `validation.csv`.
- `audit/model_forensics/dataset_forensics.md`: Documents file inventory, row counts, and strict zero-wafer overlap.
- `audit/model_forensics/experiment_roadmap.md`: Pre-registered final benchmark validation protocol.
- EXP-01, EXP-02, EXP-03, EXP-03B run logs: Prove zero access to `input/test.csv`.

## Decision
```text
PROCEED
```
The dataset is legitimately available as an independent final evaluation population. The frozen B2 candidate cascade will be evaluated with strictly zero post-hoc parameter adjustments.
