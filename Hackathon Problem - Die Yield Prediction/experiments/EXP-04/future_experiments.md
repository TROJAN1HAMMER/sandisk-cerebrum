# EXP-04: Quarantined Future Experiment Ideas

## Strict Evaluation Gate
During **EXP-04**, all model parameters, decision thresholds ($\tau_A = 0.5100, \tau_B = 0.5376$), cascade gates ($[0.295, 0.550]$), and feature representations are 100% frozen. 

Any potential performance gaps or threshold alignment opportunities observed on the independent holdout `input/test.csv` are strictly quarantined here for future post-validation engineering:

1. **Threshold Recalibration on Unseen Fabrication Runs**:
   - If the independent test set displays a defect rate shift or subtle probability calibration drift, investigate temperature scaling or isotonic regression fit on a dedicated calibration split.
   - *Constraint*: Must not be retrofitted onto EXP-04 results.
2. **Dynamic / Adaptive Escalation Gating**:
   - Explore wafer-level adaptive gating where Gate B boundaries widen or narrow based on the estimated wafer-level defect prevalence (e.g., wider gating on high-failure lots).
3. **End-to-End Deep Signal Attribution**:
   - Integrating the 1D PyTorch CNN block representation directly into the LightGBM cascade as an ensemble feature or early-exit trigger.
4. **Multi-Scale Extended Spatial Context**:
   - Formulating $7\times 7$ and $9\times 9$ spatial defect density matrices to capture large-scale scratch patterns and wafer edge ring defects.
