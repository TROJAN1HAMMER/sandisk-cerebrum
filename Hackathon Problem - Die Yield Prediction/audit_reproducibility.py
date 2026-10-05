import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix

base_dir = os.path.dirname(os.path.abspath(__file__))
val_pred_path = os.path.join(base_dir, 'results', 'validation_predictions.csv')

if not os.path.exists(val_pred_path):
    print(f"Error: {val_pred_path} not found!")
    exit(1)

print(f"Loading {val_pred_path} for Independent Metric Audit...")
df = pd.read_csv(val_pred_path)

# Verify Eligible Die Filtering
print(f"Total Validation Dies in CSV: {len(df):,}")
if 'old_label' in df.columns:
    assert (df['old_label'] == 0).all(), "Data Leakage Warning: Non-eligible dies (old_label == 1) found in validation predictions!"
    print("[PASS]: All dies in validation predictions are eligible pre-test passing dies (old_label == 0).")

y_true = df['label'].values
p_A = df['model_A_prob'].values
p_cascade = df['cascade_final_prob'].values
pred_A = df['model_A_decision'].values
pred_cascade = df['cascade_final_decision'].values

# Model A Metrics
f1_A = f1_score(y_true, pred_A)
prec_A = precision_score(y_true, pred_A)
rec_A = recall_score(y_true, pred_A)
roc_A = roc_auc_score(y_true, p_A)
p_curve_a, r_curve_a, _ = precision_recall_curve(y_true, p_A)
auc_pr_A = auc(r_curve_a, p_curve_a)
tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_true, pred_A).ravel()

# Cascade Metrics
f1_c = f1_score(y_true, pred_cascade)
prec_c = precision_score(y_true, pred_cascade)
rec_c = recall_score(y_true, pred_cascade)
roc_c = roc_auc_score(y_true, p_cascade)
p_curve_c, r_curve_c, _ = precision_recall_curve(y_true, p_cascade)
auc_pr_c = auc(r_curve_c, p_curve_c)
tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_true, pred_cascade).ravel()

# FN Recovery & Escalation Metrics
fn_mask_A = (y_true == 1) & (pred_A == 0)
fn_total_A = fn_mask_A.sum()
fn_recovered = (pred_cascade[fn_mask_A] == 1).sum()
fn_recovery_pct = (fn_recovered / fn_total_A) * 100 if fn_total_A > 0 else 0
new_fp = fp_c - fp_a

escalated_mask = (p_A >= 0.30) & (p_A <= 0.60)
escalated_count = escalated_mask.sum()
escalated_pct = (escalated_count / len(df)) * 100

print("\n==================================================")
print("INDEPENDENT REPRODUCIBILITY AUDIT RESULTS")
print("==================================================")
print(f"Model A Baseline:")
print(f"  • Fail F1:    {f1_A:.4f}")
print(f"  • Precision:  {prec_A*100:.2f}%")
print(f"  • Recall:     {rec_A*100:.2f}%")
print(f"  • AUC-PR:     {auc_pr_A:.4f}")
print(f"  • ROC-AUC:    {roc_A:.4f}")
print(f"  • Confusion:  TP={tp_a}, FN={fn_a}, FP={fp_a}, TN={tn_a}")

print(f"\nSelective Cascade (Gate [0.30, 0.60]):")
print(f"  • Fail F1:    {f1_c:.4f} (ΔF1 = {f1_c - f1_A:+.4f})")
print(f"  • Precision:  {prec_c*100:.2f}% (ΔPrec = {(prec_c - prec_A)*100:+.2f}%)")
print(f"  • Recall:     {rec_c*100:.2f}% (ΔRec = {(rec_c - rec_A)*100:+.2f}%)")
print(f"  • AUC-PR:     {auc_pr_c:.4f}")
print(f"  • ROC-AUC:    {roc_c:.4f}")
print(f"  • Confusion:  TP={tp_c}, FN={fn_c}, FP={fp_c}, TN={tn_c}")
print(f"  • FN Recovery: {fn_recovered} / {fn_total_A} ({fn_recovery_pct:.2f}%)")
print(f"  • New False Positives: {new_fp}")
print(f"  • Escalated to 1D CNN: {escalated_count:,} / {len(df):,} ({escalated_pct:.2f}%)")
