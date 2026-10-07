#!/usr/bin/env python3
"""
Common standardized evaluation utilities for EXP-05 through EXP-09.
Ensures uniform metric calculations, wafer-level analysis, and rescue mechanics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss
)

def calculate_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idxs = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        b_mask = (bin_idxs == b)
        if b_mask.sum() > 0:
            avg_p = float(y_prob[b_mask].mean())
            emp_r = float(y_true[b_mask].mean())
            ece += (b_mask.sum() / len(y_true)) * abs(avg_p - emp_r)
    return float(ece)

def evaluate_predictions(y_true, y_prob, threshold, name="Model"):
    """Calculate complete standardized classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(rec_arr, prec_arr))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f2 = float((5 * prec * rec) / (4 * prec + rec)) if (4 * prec + rec) > 0 else 0.0
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    mcc = float(matthews_corrcoef(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(calculate_ece(y_true, y_prob))
    
    return {
        'model_name': name,
        'threshold': round(float(threshold), 4),
        'accuracy': round(acc, 5),
        'balanced_accuracy': round(bal_acc, 5),
        'precision': round(prec, 5),
        'recall': round(rec, 5),
        'f1': round(f1, 5),
        'f2': round(f2, 5),
        'mcc': round(mcc, 5),
        'pr_auc': round(pr_auc, 5),
        'roc_auc': round(roc_auc, 5),
        'specificity': round(spec, 5),
        'fpr': round(fpr, 5),
        'fnr': round(fnr, 5),
        'brier': round(brier, 5),
        'ece': round(ece, 5),
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn)
    }

def evaluate_cascade_system(
    p_A,
    p_B,
    y_true,
    tau_A=0.5100,
    tau_B=0.5376,
    gate_lower=0.295,
    gate_upper=0.550,
    name="Cascade"
):
    """
    Evaluate the complete selective cascade system.
    Returns: cascade_metrics, rescue_mechanics, casc_probs, casc_preds.
    """
    n = len(y_true)
    pred_A = (p_A >= tau_A).astype(int)
    pred_B = (p_B >= tau_B).astype(int)
    
    esc_mask = (p_A >= gate_lower) & (p_A <= gate_upper)
    esc_count = int(esc_mask.sum())
    esc_rate = float(esc_count / n)
    compute_saved = float(1.0 - esc_rate)
    
    casc_probs = np.where(esc_mask, p_B, p_A)
    casc_preds = np.where(esc_mask, pred_B, pred_A)
    
    tn, fp, fn, tp = confusion_matrix(y_true, casc_preds).ravel()
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, casc_probs)
    pr_auc = float(auc(rec_arr, prec_arr))
    roc_auc = float(roc_auc_score(y_true, casc_probs))
    
    f1 = float(f1_score(y_true, casc_preds, zero_division=0))
    prec = float(precision_score(y_true, casc_preds, zero_division=0))
    rec = float(recall_score(y_true, casc_preds, zero_division=0))
    f2 = float((5 * prec * rec) / (4 * prec + rec)) if (4 * prec + rec) > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, casc_preds))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    mcc = float(matthews_corrcoef(y_true, casc_preds))
    brier = float(brier_score_loss(y_true, casc_probs))
    ece = float(calculate_ece(y_true, casc_probs))
    
    # Rescue Mechanics Forensics
    ma_fn_mask = (pred_A == 0) & (y_true == 1)
    ma_tp_mask = (pred_A == 1) & (y_true == 1)
    ma_fp_mask = (pred_A == 1) & (y_true == 0)
    
    total_ma_fn = int(ma_fn_mask.sum())
    fn_sent = int((ma_fn_mask & esc_mask).sum())
    fn_rescued = int(((ma_fn_mask & esc_mask) & (pred_B == 1)).sum())
    fn_missed = fn_sent - fn_rescued
    rescue_rate_pct = float(fn_rescued / fn_sent * 100.0) if fn_sent > 0 else 0.0
    
    tp_sent = int((ma_tp_mask & esc_mask).sum())
    tp_lost = int(((ma_tp_mask & esc_mask) & (pred_B == 0)).sum())
    tp_preserved = tp_sent - tp_lost
    
    fp_sent = int((ma_fp_mask & esc_mask).sum())
    fp_removed = int(((ma_fp_mask & esc_mask) & (pred_B == 0)).sum())
    new_fp = int(((~ma_fp_mask & (y_true == 0) & esc_mask) & (pred_B == 1)).sum())
    net_tp_gain = fn_rescued - tp_lost
    
    cascade_metrics = {
        'system_name': name,
        'tau_A': round(float(tau_A), 4),
        'tau_B': round(float(tau_B), 4),
        'gate_lower': round(float(gate_lower), 4),
        'gate_upper': round(float(gate_upper), 4),
        'f1': round(f1, 5),
        'f2': round(f2, 5),
        'precision': round(prec, 5),
        'recall': round(rec, 5),
        'mcc': round(mcc, 5),
        'pr_auc': round(pr_auc, 5),
        'roc_auc': round(roc_auc, 5),
        'balanced_accuracy': round(bal_acc, 5),
        'specificity': round(spec, 5),
        'fpr': round(fpr, 5),
        'brier': round(brier, 5),
        'ece': round(ece, 5),
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn),
        'escalation_count': int(esc_count),
        'escalation_rate_pct': round(esc_rate * 100.0, 4),
        'compute_saved_pct': round(compute_saved * 100.0, 4)
    }
    
    rescue_mechanics = {
        'model_A_fn_total': int(total_ma_fn),
        'fn_escalated': int(fn_sent),
        'fn_rescued': int(fn_rescued),
        'fn_missed': int(fn_missed),
        'rescue_rate_pct': round(rescue_rate_pct, 2),
        'model_A_tp_escalated': int(tp_sent),
        'tp_lost': int(tp_lost),
        'tp_preserved': int(tp_preserved),
        'net_tp_gain': int(net_tp_gain),
        'new_fp_introduced': int(new_fp)
    }
    
    return cascade_metrics, rescue_mechanics, casc_probs, casc_preds

def compute_wafer_robustness_summary(wafer_ids, y_true, baseline_preds, candidate_preds):
    """Compute per-wafer F1 and comparison against baseline."""
    unique_wafers = np.unique(wafer_ids)
    records = []
    
    for wid in unique_wafers:
        mask = (wafer_ids == wid)
        y_w = y_true[mask]
        b_w = baseline_preds[mask]
        c_w = candidate_preds[mask]
        
        n_dies = int(mask.sum())
        n_def = int(y_w.sum())
        
        f1_b = float(f1_score(y_w, b_w, zero_division=0)) if n_def > 0 else 1.0
        f1_c = float(f1_score(y_w, c_w, zero_division=0)) if n_def > 0 else 1.0
        delta = f1_c - f1_b
        
        if delta > 0.005:
            status = 'IMPROVED'
        elif delta < -0.005:
            status = 'DEGRADED'
        else:
            status = 'TIED'
            
        records.append({
            'wafer_id': wid,
            'dies': n_dies,
            'defects': n_def,
            'baseline_f1': round(f1_b, 4),
            'candidate_f1': round(f1_c, 4),
            'delta_f1': round(delta, 4),
            'status': status
        })
        
    df_w = pd.DataFrame(records)
    summary = {
        'total_wafers': len(unique_wafers),
        'mean_baseline_f1': round(float(df_w['baseline_f1'].mean()), 4),
        'mean_candidate_f1': round(float(df_w['candidate_f1'].mean()), 4),
        'median_baseline_f1': round(float(df_w['baseline_f1'].median()), 4),
        'median_candidate_f1': round(float(df_w['candidate_f1'].median()), 4),
        'std_candidate_f1': round(float(df_w['candidate_f1'].std()), 4),
        'improved_wafers': int((df_w['status'] == 'IMPROVED').sum()),
        'tied_wafers': int((df_w['status'] == 'TIED').sum()),
        'degraded_wafers': int((df_w['status'] == 'DEGRADED').sum())
    }
    return df_w, summary
