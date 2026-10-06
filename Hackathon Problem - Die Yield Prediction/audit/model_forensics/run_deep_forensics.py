"""
Deep Forensics Analysis Script for SanDisk Cerebrum.
Performs in-depth wafer-by-wafer, calibration, threshold, signal, and spatial forensic computations.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, auc, brier_score_loss
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
AUDIT_DIR = os.path.join(BASE_DIR, 'audit', 'model_forensics')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

diag_path = os.path.join(RESULTS_DIR, 'val_die_diagnostics.csv')
df = pd.read_csv(diag_path)

# Cascade predictions under Gate B [0.25, 0.55]
escalated = (df['prob_Model_A'] >= 0.25) & (df['prob_Model_A'] <= 0.55)
df['decision_Cascade'] = np.where(escalated, df['decision_Model_B'], df['decision_Model_A'])
df['prob_Cascade'] = np.where(escalated, df['prob_Model_B'], df['prob_Model_A'])

print("=== 1. WAFER-LEVEL FORENSICS (32 WAFERS) ===")
wafer_records = []
for wid, w_df in df.groupby('wafer_id'):
    y_w = w_df['actual_label'].values
    p_A = w_df['decision_Model_A'].values
    p_Cas = w_df['decision_Cascade'].values
    
    samples = len(w_df)
    fails = int(y_w.sum())
    yield_pct = round((samples - fails) / samples * 100, 2)
    
    tp_A = int(((y_w == 1) & (p_A == 1)).sum())
    fn_A = int(((y_w == 1) & (p_A == 0)).sum())
    fp_A = int(((y_w == 0) & (p_A == 1)).sum())
    tn_A = int(((y_w == 0) & (p_A == 0)).sum())
    rec_A = round(tp_A / fails, 4) if fails > 0 else 0.0
    prec_A = round(tp_A / (tp_A + fp_A), 4) if (tp_A + fp_A) > 0 else 0.0
    f1_A = round(2 * prec_A * rec_A / (prec_A + rec_A), 4) if (prec_A + rec_A) > 0 else 0.0
    
    tp_C = int(((y_w == 1) & (p_Cas == 1)).sum())
    fn_C = int(((y_w == 1) & (p_Cas == 0)).sum())
    fp_C = int(((y_w == 0) & (p_Cas == 1)).sum())
    tn_C = int(((y_w == 0) & (p_Cas == 0)).sum())
    rec_C = round(tp_C / fails, 4) if fails > 0 else 0.0
    prec_C = round(tp_C / (tp_C + fp_C), 4) if (tp_C + fp_C) > 0 else 0.0
    f1_C = round(2 * prec_C * rec_C / (prec_C + rec_C), 4) if (prec_C + rec_C) > 0 else 0.0
    
    delta_tp = tp_C - tp_A
    delta_fp = fp_C - fp_A
    delta_f1 = round(f1_C - f1_A, 4)
    
    wafer_records.append({
        'wafer_id': wid,
        'samples': samples,
        'actual_failures': fails,
        'yield_pct': yield_pct,
        'model_A_tp': tp_A,
        'model_A_fn': fn_A,
        'model_A_fp': fp_A,
        'model_A_recall': rec_A,
        'model_A_precision': prec_A,
        'model_A_f1': f1_A,
        'cascade_tp': tp_C,
        'cascade_fn': fn_C,
        'cascade_fp': fp_C,
        'cascade_recall': rec_C,
        'cascade_precision': prec_C,
        'cascade_f1': f1_C,
        'delta_tp': delta_tp,
        'delta_fp': delta_fp,
        'delta_f1': delta_f1
    })

wafer_df = pd.DataFrame(wafer_records)
wafer_csv_path = os.path.join(AUDIT_DIR, 'wafer_forensics_breakdown.csv')
wafer_df.to_csv(wafer_csv_path, index=False)
print(f"Saved wafer forensics breakdown to {wafer_csv_path}")

# Print summary wafer metrics
print(f"Top 3 Wafers with Most Cascade Benefit (ΔF1):")
print(wafer_df.sort_values('delta_f1', ascending=False)[['wafer_id', 'actual_failures', 'model_A_recall', 'cascade_recall', 'delta_tp', 'delta_fp', 'delta_f1']].head(3).to_string(index=False))

print(f"\nWafers where Cascade Hurts (ΔF1 < 0):")
print(wafer_df[wafer_df['delta_f1'] < 0][['wafer_id', 'actual_failures', 'model_A_recall', 'cascade_recall', 'delta_tp', 'delta_fp', 'delta_f1']].to_string(index=False))

# --- CALIBRATION FORENSICS ---
print("\n=== 2. CALIBRATION & RELIABILITY FORENSICS ===")
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
cal_records = []
for i in range(len(bins)-1):
    b_low, b_high = bins[i], bins[i+1]
    if b_high == 1.0:
        sub_A = df[(df['prob_Model_A'] >= b_low) & (df['prob_Model_A'] <= b_high)]
        sub_C = df[(df['prob_Cascade'] >= b_low) & (df['prob_Cascade'] <= b_high)]
    else:
        sub_A = df[(df['prob_Model_A'] >= b_low) & (df['prob_Model_A'] < b_high)]
        sub_C = df[(df['prob_Cascade'] >= b_low) & (df['prob_Cascade'] < b_high)]
        
    cal_records.append({
        'bin_range': f"[{b_low:.1f}, {b_high:.1f})",
        'model_A_dies': len(sub_A),
        'model_A_avg_prob': round(sub_A['prob_Model_A'].mean(), 4) if len(sub_A) > 0 else 0,
        'model_A_empirical_rate': round(sub_A['actual_label'].mean(), 4) if len(sub_A) > 0 else 0,
        'cascade_dies': len(sub_C),
        'cascade_avg_prob': round(sub_C['prob_Cascade'].mean(), 4) if len(sub_C) > 0 else 0,
        'cascade_empirical_rate': round(sub_C['actual_label'].mean(), 4) if len(sub_C) > 0 else 0,
    })

cal_df = pd.DataFrame(cal_records)
print(cal_df.to_string(index=False))

# --- THRESHOLD FORENSICS ---
print("\n=== 3. THRESHOLD LANDSCAPE AUDIT ===")
thresh_grid = np.arange(0.10, 0.90, 0.05)
thresh_res = []
y_val = df['actual_label'].values
p_val_A = df['prob_Model_A'].values
for t in thresh_grid:
    pred = (p_val_A >= t).astype(int)
    f1 = f1_score(y_val, pred, zero_division=0)
    rec = recall_score(y_val, pred, zero_division=0)
    prec = precision_score(y_val, pred, zero_division=0)
    f2 = (5*prec*rec)/(4*prec+rec) if (4*prec+rec)>0 else 0
    cm = confusion_matrix(y_val, pred).ravel()
    thresh_res.append({
        'threshold': round(t, 2),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'f2': round(f2, 4),
        'tp': cm[3], 'fp': cm[1], 'fn': cm[2], 'tn': cm[0]
    })
t_df = pd.DataFrame(thresh_res)
opt_f1 = t_df.loc[t_df['f1'].idxmax()]
opt_f2 = t_df.loc[t_df['f2'].idxmax()]
print(f"Optimal F1 on Model A: Thresh = {opt_f1['threshold']}, F1 = {opt_f1['f1']}, Recall = {opt_f1['recall']}, Precision = {opt_f1['precision']}")
print(f"Optimal F2 on Model A: Thresh = {opt_f2['threshold']}, F2 = {opt_f2['f2']}, Recall = {opt_f2['recall']}, Precision = {opt_f2['precision']}")

# --- SIGNAL & SPATIAL CORRELATION ---
print("\n=== 4. SIGNAL & SPATIAL CORRELATION FORENSICS ===")
corr_cols = ['radial_dist', 'density_3x3', 'density_5x5', 'edge_dist', 'b_mean', 'b_std', 'b_max', 'cnt_104', 'longest_run_104', 'cluster_area']
corrs = df[corr_cols].apply(lambda c: c.corr(df['actual_label']))
corr_df = pd.DataFrame({'feature': corrs.index, 'correlation_with_actual_label': corrs.values}).sort_values('correlation_with_actual_label', ascending=False)
print(corr_df.to_string(index=False))

# --- COHORT STATISTICAL COMPARISON ---
print("\n=== 5. ERROR COHORT STATISTICAL PROFILE ===")
cohort_summary = []
for c_name, mask in [('FN (699)', (df['actual_label'] == 1) & (df['decision_Model_A'] == 0)),
                     ('FP (42)', (df['actual_label'] == 0) & (df['decision_Model_A'] == 1)),
                     ('TP (357)', (df['actual_label'] == 1) & (df['decision_Model_A'] == 1)),
                     ('TN (26874)', (df['actual_label'] == 0) & (df['decision_Model_A'] == 0))]:
    sub = df[mask]
    cohort_summary.append({
        'cohort': c_name,
        'count': len(sub),
        'prob_A_mean': round(sub['prob_Model_A'].mean(), 4),
        'prob_A_median': round(sub['prob_Model_A'].median(), 4),
        'prob_B_mean': round(sub['prob_Model_B'].mean(), 4),
        'radial_dist_mean': round(sub['radial_dist'].mean(), 4),
        'density_3x3_mean': round(sub['density_3x3'].mean(), 4),
        'b_mean_mean': round(sub['b_mean'].mean(), 2),
        'cnt_104_mean': round(sub['cnt_104'].mean(), 2),
        'cluster_area_mean': round(sub['cluster_area'].mean(), 2)
    })
print(pd.DataFrame(cohort_summary).to_string(index=False))
