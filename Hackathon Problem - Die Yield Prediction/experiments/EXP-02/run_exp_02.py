#!/usr/bin/env python3
"""
EXP-02: Joint Threshold & Asymmetric Cascade Optimization
SanDisk Cerebrum Die Yield Prediction Platform

Scientific Evaluation & Decision Policy Optimization
DO NOT RETRAIN MODELS. DO NOT OVERWRITE PRODUCTION ARTIFACTS.
"""

import os
import sys
import time
import json
import hashlib
import pickle
import platform
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXP_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
INPUT_DIR = os.path.join(PROJECT_ROOT, 'input')
EXP01_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'EXP-01')
PLOTS_DIR = os.path.join(EXP_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Add PROJECT_ROOT to path
sys.path.insert(0, PROJECT_ROOT)
from src.spatial import compute_spatial_features

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def compute_sha256(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def calculate_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idxs = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        b_mask = (bin_idxs == b)
        if b_mask.sum() > 0:
            avg_p = y_prob[b_mask].mean()
            emp_r = y_true[b_mask].mean()
            ece += (b_mask.sum() / len(y_true)) * abs(avg_p - emp_r)
    return float(ece)

# -------------------------------------------------------------
# 1. INTEGRITY PRE-CHECK
# -------------------------------------------------------------
log("=" * 80)
log("EXP-02: Joint Threshold & Asymmetric Cascade Optimization")
log("=" * 80)

FILES_TO_LOCK = [
    'models/model_A.txt',
    'models/model_A.pkl',
    'models/model_B.pkl',
    'models/cnn_1d_block_model.pt',
    'models/selective_cascade_meta.pkl',
    'models/feature_metadata.json',
    'results/AUTHORITATIVE_CLEAN_METRICS.json',
    'results/val_die_diagnostics.csv',
    'results/canonical_wafer_split.json'
]

hashes_before_path = os.path.join(EXP_DIR, 'hashes_before.json')
if not os.path.exists(hashes_before_path):
    hashes_before = {}
    for rel_path in FILES_TO_LOCK:
        full_p = os.path.join(PROJECT_ROOT, rel_path)
        hashes_before[rel_path] = compute_sha256(full_p)
    with open(hashes_before_path, 'w') as f:
        json.dump(hashes_before, f, indent=2)
else:
    with open(hashes_before_path, 'r') as f:
        hashes_before = json.load(f)

log("Verifying pre-experiment file hashes...")
for rel_path, expected_hash in hashes_before.items():
    current_hash = compute_sha256(os.path.join(PROJECT_ROOT, rel_path))
    assert current_hash == expected_hash, f"PRE-RUN MUTATION DETECTED in {rel_path}!"
log("✅ Pre-experiment hashes verified 100% bit-for-bit identical.")

# -------------------------------------------------------------
# 2. LOAD DATASET & VERIFY EXP-01 ARTIFACT
# -------------------------------------------------------------
log("\n[1/7] Loading canonical holdout validation set and model predictions...")

cache_npz = os.path.join(EXP_DIR, 'cached_val_data.npz')
if os.path.exists(cache_npz):
    log("  • Loading cached validation vectors from cached_val_data.npz...")
    cached = np.load(cache_npz, allow_pickle=True)
    val_probs_exp = cached['val_probs_exp']
    val_probs_base = cached['val_probs_base']
    val_probs_B = cached['val_probs_B']
    y_val = cached['y_val']
    wafer_ids = cached['wafer_ids']
    die_rows = cached['die_rows']
    die_cols = cached['die_cols']
else:
    log("  • Computing validation features and unrounded predictions...")
    with open(os.path.join(RESULTS_DIR, 'canonical_wafer_split.json'), 'r') as f:
        split = json.load(f)
    val_wafers = set(split['validation_wafers'])

    df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
    eligible = df[df['old_label'] == 0].copy().reset_index(drop=True)
    val_mask = eligible['wafer_id'].isin(val_wafers)
    val_df = eligible[val_mask].reset_index(drop=True)

    sp_after_all = compute_spatial_features(eligible, full_df=df)
    sp_after_val = sp_after_all[val_mask].reset_index(drop=True)

    feat_cols = [f'feature_{i}' for i in range(1, 501)]
    spatial_names = ['radial_dist', 'density_3x3', 'density_5x5', 'edge_dist', 'die_row', 'die_col']
    val_A_exp_input = pd.concat([val_df[feat_cols], sp_after_val[spatial_names]], axis=1)

    exp_pkl_path = os.path.join(EXP01_DIR, 'model_A_spatial.pkl')
    with open(exp_pkl_path, 'rb') as f:
        clf_exp = pickle.load(f)

    val_probs_exp = clf_exp.predict_proba(val_A_exp_input)[:, 1]

    diag_df = pd.read_csv(os.path.join(RESULTS_DIR, 'val_die_diagnostics.csv'))
    assert (diag_df['wafer_id'].values == val_df['wafer_id'].values).all()
    assert (diag_df['die_row'].values == val_df['die_row'].values).all()
    assert (diag_df['die_col'].values == val_df['die_col'].values).all()

    val_probs_base = diag_df['prob_Model_A'].values
    val_probs_B = diag_df['prob_Model_B'].values
    y_val = diag_df['actual_label'].values
    wafer_ids = diag_df['wafer_id'].values
    die_rows = diag_df['die_row'].values
    die_cols = diag_df['die_col'].values

    np.savez_compressed(
        cache_npz,
        val_probs_exp=val_probs_exp,
        val_probs_base=val_probs_base,
        val_probs_B=val_probs_B,
        y_val=y_val,
        wafer_ids=wafer_ids,
        die_rows=die_rows,
        die_cols=die_cols
    )
    log(f"  • Cached validation arrays to {cache_npz}")

N_VAL = len(y_val)
N_FAIL = int(y_val.sum())
log(f"  • Validation dies: {N_VAL:,} across {len(np.unique(wafer_ids))} wafers | Failures: {N_FAIL:,} ({N_FAIL/N_VAL*100:.4f}%)")

# Confirm baseline replication:
val_preds_B = (val_probs_B >= 0.5376).astype(int)

# 1. Production baseline check:
prod_esc = (val_probs_base >= 0.25) & (val_probs_base <= 0.55)
prod_preds = np.where(prod_esc, val_preds_B, (val_probs_base >= 0.5100).astype(int))
prod_probs = np.where(prod_esc, val_probs_B, val_probs_base)
cm_prod = confusion_matrix(y_val, prod_preds).ravel()
log(f"  • Prod Baseline Cascade: TP={cm_prod[3]}, FN={cm_prod[2]}, FP={cm_prod[1]}, TN={cm_prod[0]}, Esc={prod_esc.sum()} ({prod_esc.mean()*100:.2f}%)")
assert cm_prod[3] == 389 and cm_prod[1] == 89, "Baseline cascade mismatch!"

# 2. EXP-01 locked check:
exp01_esc = (val_probs_exp >= 0.25) & (val_probs_exp <= 0.55)
exp01_preds = np.where(exp01_esc, val_preds_B, (val_probs_exp >= 0.5100).astype(int))
exp01_probs = np.where(exp01_esc, val_probs_B, val_probs_exp)
cm_exp01 = confusion_matrix(y_val, exp01_preds).ravel()
log(f"  • EXP-01 Locked Cascade:  TP={cm_exp01[3]}, FN={cm_exp01[2]}, FP={cm_exp01[1]}, TN={cm_exp01[0]}, Esc={exp01_esc.sum()} ({exp01_esc.mean()*100:.2f}%)")
assert cm_exp01[3] == 394 and cm_exp01[1] == 83, "EXP-01 cascade mismatch!"
log("✅ Both reference cascades matched bit-for-bit!")

# -------------------------------------------------------------
# 3. FULL DETERMINISTIC GRID SEARCH
# -------------------------------------------------------------
log("\n[2/7] Running full deterministic grid search across tau_A x gate_lower...")

tau_A_values = np.round(np.arange(0.40, 0.5501, 0.005), 4)      # 31 values
gate_lower_values = np.round(np.arange(0.15, 0.3001, 0.005), 4) # 31 values
LOCKED_GATE_UPPER = 0.55
LOCKED_TAU_B = 0.5376

log(f"  • tau_A grid: {len(tau_A_values)} points from {tau_A_values[0]:.3f} to {tau_A_values[-1]:.3f} (step 0.005)")
log(f"  • gate_lower grid: {len(gate_lower_values)} points from {gate_lower_values[0]:.3f} to {gate_lower_values[-1]:.3f} (step 0.005)")
log(f"  • Total operating points: {len(tau_A_values) * len(gate_lower_values):,} evaluations")

# Precompute gate masks and gate metrics for each gate_lower (since probs & ranking depend solely on gate_lower)
gate_cache = {}
for gl in gate_lower_values:
    esc_mask = (val_probs_exp >= gl) & (val_probs_exp <= LOCKED_GATE_UPPER)
    casc_probs = np.where(esc_mask, val_probs_B, val_probs_exp)
    
    # Ranking metrics
    prec_arr, rec_arr, _ = precision_recall_curve(y_val, casc_probs)
    pr_auc = auc(rec_arr, prec_arr)
    roc_auc = roc_auc_score(y_val, casc_probs)
    brier = brier_score_loss(y_val, casc_probs)
    ece = calculate_ece(y_val, casc_probs)
    
    esc_count = int(esc_mask.sum())
    esc_rate = esc_count / N_VAL
    compute_saved = 1.0 - esc_rate
    
    gate_cache[gl] = {
        'esc_mask': esc_mask,
        'casc_probs': casc_probs,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'brier': brier,
        'ece': ece,
        'esc_count': esc_count,
        'esc_rate': esc_rate,
        'compute_saved': compute_saved
    }

grid_records = []
for tau in tau_A_values:
    # Model A standalone predictions at this tau
    pred_A_standalone = (val_probs_exp >= tau).astype(int)
    cm_A = confusion_matrix(y_val, pred_A_standalone).ravel()
    tp_A_alone = cm_A[3]
    fn_A_alone = cm_A[2]
    fp_A_alone = cm_A[1]
    
    for gl in gate_lower_values:
        gc = gate_cache[gl]
        esc_mask = gc['esc_mask']
        casc_probs = gc['casc_probs']
        
        # Cascade predictions
        casc_preds = np.where(esc_mask, val_preds_B, pred_A_standalone)
        
        tn, fp, fn, tp = confusion_matrix(y_val, casc_preds).ravel()
        
        acc = (tp + tn) / N_VAL
        bal_acc = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
        
        denom_mcc = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (float(tp * tn) - float(fp * fn)) / denom_mcc if denom_mcc > 0 else 0.0
        
        # Operational accounting
        # Model A FN sent to B and rescued
        ma_fn_mask = (pred_A_standalone == 0) & (y_val == 1)
        gross_fn_rescued = int(((ma_fn_mask & esc_mask) & (val_preds_B == 1)).sum())
        
        # Model A TP sent to B and lost
        ma_tp_mask = (pred_A_standalone == 1) & (y_val == 1)
        tp_lost = int(((ma_tp_mask & esc_mask) & (val_preds_B == 0)).sum())
        
        net_tp_gain = gross_fn_rescued - tp_lost
        
        # Model A FP sent to B and eliminated / new FPs introduced
        ma_fp_mask = (pred_A_standalone == 1) & (y_val == 0)
        fp_removed = int(((ma_fp_mask & esc_mask) & (val_preds_B == 0)).sum())
        ma_passing_mask = (pred_A_standalone == 0) & (y_val == 0)
        new_fp_introduced = int(((ma_passing_mask & esc_mask) & (val_preds_B == 1)).sum())
        
        rec_data = {
            'tau_A': tau,
            'gate_lower': gl,
            'gate_upper': LOCKED_GATE_UPPER,
            'tau_B': LOCKED_TAU_B,
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp),
            'tn': int(tn),
            'accuracy': round(acc, 5),
            'balanced_accuracy': round(bal_acc, 5),
            'precision': round(prec, 5),
            'recall': round(rec, 5),
            'specificity': round(spec, 5),
            'fpr': round(fpr, 5),
            'fnr': round(fnr, 5),
            'f1': round(f1, 5),
            'f2': round(f2, 5),
            'mcc': round(mcc, 5),
            'pr_auc': round(gc['pr_auc'], 5),
            'roc_auc': round(gc['roc_auc'], 5),
            'brier': round(gc['brier'], 5),
            'ece': round(gc['ece'], 5),
            'total_dies': N_VAL,
            'fast_path_dies': N_VAL - gc['esc_count'],
            'escalated_dies': gc['esc_count'],
            'escalation_rate': round(gc['esc_rate'], 5),
            'compute_saved': round(gc['compute_saved'], 5),
            'gross_fn_recovery': gross_fn_rescued,
            'tp_lost': tp_lost,
            'net_tp_gain': net_tp_gain,
            'fp_removed': fp_removed,
            'new_fp_introduced': new_fp_introduced,
            'defect_escape_rate': round(fnr, 5),
            'false_alarm_rate': round(fpr, 5)
        }
        grid_records.append(rec_data)

grid_df = pd.DataFrame(grid_records)
grid_df.to_csv(os.path.join(EXP_DIR, 'grid_results.csv'), index=False)
log(f"  • Grid evaluation completed: {len(grid_df)} rows saved to grid_results.csv")

# -------------------------------------------------------------
# 4. SECONDARY SWEEP CHECK: GATE UPPER BOUND
# -------------------------------------------------------------
log("\n[3/7] Evaluating secondary sweep for gate_upper around top operating region...")

# Find top operating points by F1 and F2 under primary constraints
best_f1_row = grid_df.loc[grid_df['f1'].idxmax()]
tau_top = best_f1_row['tau_A']
gl_top = best_f1_row['gate_lower']

log(f"  • Top grid point for secondary probe: tau_A={tau_top:.3f}, gate_lower={gl_top:.3f}")

gate_upper_values = np.round(np.arange(0.50, 0.6501, 0.01), 2)
sec_records = []
for gu in gate_upper_values:
    esc_mask = (val_probs_exp >= gl_top) & (val_probs_exp <= gu)
    casc_probs = np.where(esc_mask, val_probs_B, val_probs_exp)
    casc_preds = np.where(esc_mask, val_preds_B, (val_probs_exp >= tau_top).astype(int))
    
    tn, fp, fn, tp = confusion_matrix(y_val, casc_preds).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    
    denom_mcc = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (float(tp * tn) - float(fp * fn)) / denom_mcc if denom_mcc > 0 else 0.0
    
    prec_arr, rec_arr, _ = precision_recall_curve(y_val, casc_probs)
    pr_auc = auc(rec_arr, prec_arr)
    
    sec_records.append({
        'gate_upper': gu,
        'tau_A': tau_top,
        'gate_lower': gl_top,
        'f1': round(f1, 5),
        'f2': round(f2, 5),
        'recall': round(rec, 5),
        'precision': round(prec, 5),
        'mcc': round(mcc, 5),
        'pr_auc': round(pr_auc, 5),
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'escalation_rate': round(esc_mask.mean(), 5),
        'compute_saved': round(1.0 - esc_mask.mean(), 5)
    })
sec_df = pd.DataFrame(sec_records)
sec_df.to_csv(os.path.join(EXP_DIR, 'gate_upper_sweep.csv'), index=False)
log("  • gate_upper secondary sweep results:")
log(sec_df[['gate_upper', 'f1', 'f2', 'recall', 'precision', 'fp', 'escalation_rate']].to_string(index=False))

# -------------------------------------------------------------
# 5. PARETO FRONTIER & CANDIDATE OPERATING POINTS
# -------------------------------------------------------------
log("\n[4/7] Computing multi-objective Pareto frontier and candidate operating points...")

# Objectives to maximize: recall, precision, f1, f2, mcc, pr_auc, compute_saved
# Objectives to minimize: fp, fn, escalation_rate
# A point i is dominated by j if j is >= on all maximization objectives, <= on all minimization objectives,
# and strictly better on at least one.

pareto_mask = []
for i, row_i in grid_df.iterrows():
    dominated = False
    for j, row_j in grid_df.iterrows():
        if i == j:
            continue
        # Compare row_j against row_i
        # Better or equal on all:
        ge_all = (
            (row_j['recall'] >= row_i['recall']) and
            (row_j['precision'] >= row_i['precision']) and
            (row_j['f1'] >= row_i['f1']) and
            (row_j['f2'] >= row_i['f2']) and
            (row_j['mcc'] >= row_i['mcc']) and
            (row_j['pr_auc'] >= row_i['pr_auc']) and
            (row_j['compute_saved'] >= row_i['compute_saved']) and
            (row_j['fp'] <= row_i['fp']) and
            (row_j['fn'] <= row_i['fn'])
        )
        # Strictly better on at least one:
        strictly_better = (
            (row_j['recall'] > row_i['recall']) or
            (row_j['precision'] > row_i['precision']) or
            (row_j['f1'] > row_i['f1']) or
            (row_j['f2'] > row_i['f2']) or
            (row_j['mcc'] > row_i['mcc']) or
            (row_j['pr_auc'] > row_i['pr_auc']) or
            (row_j['compute_saved'] > row_i['compute_saved']) or
            (row_j['fp'] < row_i['fp']) or
            (row_j['fn'] < row_i['fn'])
        )
        if ge_all and strictly_better:
            dominated = True
            break
    pareto_mask.append(not dominated)

pareto_df = grid_df[pareto_mask].copy().sort_values(by='f1', ascending=False)
pareto_df.to_csv(os.path.join(EXP_DIR, 'pareto_frontier.csv'), index=False)
log(f"  • Pareto frontier identified: {len(pareto_df)} non-dominated operating points out of {len(grid_df)}")

# Candidate Operating Points Extraction
# Use distance to tau_A=0.5100 to break ties (since cascade metrics are invariant across tau_A within [gate_lower, gate_upper])
grid_df['tau_dist'] = (grid_df['tau_A'] - 0.5100).abs()

# 1. Best Recall under FP <= 100
cand_best_rec_fp100 = grid_df[grid_df['fp'] <= 100].sort_values(by=['recall', 'f1', 'tau_dist'], ascending=[False, False, True]).iloc[0]

# 2. Best Recall under Escalation <= 15%
cand_best_rec_esc15 = grid_df[grid_df['escalation_rate'] <= 0.15].sort_values(by=['recall', 'f1', 'tau_dist'], ascending=[False, False, True]).iloc[0]

# 3. Best F2 under FP <= 100
cand_best_f2_fp100 = grid_df[grid_df['fp'] <= 100].sort_values(by=['f2', 'recall', 'tau_dist'], ascending=[False, False, True]).iloc[0]

# 4. Best MCC under Escalation <= 15%
cand_best_mcc_esc15 = grid_df[grid_df['escalation_rate'] <= 0.15].sort_values(by=['mcc', 'f1', 'tau_dist'], ascending=[False, False, True]).iloc[0]

# 5. Best F1 under Recall >= 38%
rec_ge_38 = grid_df[grid_df['recall'] >= 0.38]
if len(rec_ge_38) > 0:
    cand_best_f1_rec38 = rec_ge_38.sort_values(by=['f1', 'mcc', 'tau_dist'], ascending=[False, False, True]).iloc[0]
else:
    cand_best_f1_rec38 = grid_df.sort_values(by=['recall', 'f1', 'tau_dist'], ascending=[False, False, True]).iloc[0]

# 6. Global Maximum F1
cand_max_f1 = grid_df.sort_values(by=['f1', 'precision', 'tau_dist'], ascending=[False, False, True]).iloc[0]

# 7. Global Maximum F2
cand_max_f2 = grid_df.sort_values(by=['f2', 'recall', 'tau_dist'], ascending=[False, False, True]).iloc[0]

# 8. Balanced High-Precision / High-F1 Operating Point
# Let's inspect primary target constraints:
# F1 >= 0.525, Recall >= 0.38, Precision >= 0.80, MCC >= 0.54, PR-AUC >= 0.475, FP <= 90, Escalation <= 15%, Compute Saved >= 85%
sat_primary = grid_df[
    (grid_df['f1'] >= 0.525) &
    (grid_df['recall'] >= 0.38) &
    (grid_df['precision'] >= 0.80) &
    (grid_df['mcc'] >= 0.54) &
    (grid_df['pr_auc'] >= 0.475) &
    (grid_df['fp'] <= 90) &
    (grid_df['escalation_rate'] <= 0.15) &
    (grid_df['compute_saved'] >= 0.85)
]
log(f"  • Operating points satisfying ALL primary target constraints: {len(sat_primary)}")

candidate_list = [
    ('MAX_F1_COMPUTE_OPTIMAL', cand_max_f1, 'PREFERRED'),
    ('BEST_RECALL_CONTROLLED_FP', cand_best_rec_fp100, 'VIABLE'),
    ('BEST_F2_BALANCED', cand_best_f2_fp100, 'VIABLE'),
    ('BEST_MCC_EFFICIENT', cand_best_mcc_esc15, 'VIABLE'),
    ('MAX_F2_AGGRESSIVE_RESCUE', cand_max_f2, 'REJECT')
]

cand_rows = []
for label, row, status in candidate_list:
    r_dict = row.to_dict()
    r_dict['candidate_id'] = label
    r_dict['status'] = status
    cand_rows.append(r_dict)

cand_df = pd.DataFrame(cand_rows)
cand_df.to_csv(os.path.join(EXP_DIR, 'candidate_operating_points.csv'), index=False)
log("  • Saved candidate operating points:")
for _, r in cand_df.iterrows():
    log(f"    [{r['status']}] {r['candidate_id']}: tau_A={r['tau_A']:.3f}, gate=[{r['gate_lower']:.3f}, {r['gate_upper']:.2f}] | F1={r['f1']:.4f}, F2={r['f2']:.4f}, Rec={r['recall']*100:.2f}%, Prec={r['precision']*100:.2f}%, TP={r['tp']}, FP={r['fp']}, Esc={r['escalation_rate']*100:.2f}%")

# Select PREFERRED candidate
preferred_row = cand_max_f1
tau_A_pref = preferred_row['tau_A']
gl_pref = preferred_row['gate_lower']
gu_pref = preferred_row['gate_upper']

# -------------------------------------------------------------
# 6. WAFER-LEVEL ROBUSTNESS ANALYSIS
# -------------------------------------------------------------
log("\n[5/7] Conducting wafer-level robustness analysis across all 32 wafers...")

# Generate predictions for:
# 1. Baseline
# 2. EXP-01 locked
# 3. EXP-02 Preferred (tau_A_pref, gl_pref)
pref_esc_mask = (val_probs_exp >= gl_pref) & (val_probs_exp <= gu_pref)
pref_preds = np.where(pref_esc_mask, val_preds_B, (val_probs_exp >= tau_A_pref).astype(int))
pref_probs = np.where(pref_esc_mask, val_probs_B, val_probs_exp)

unique_wafers = np.unique(wafer_ids)
wafer_records = []

for wid in unique_wafers:
    w_mask = (wafer_ids == wid)
    y_w = y_val[w_mask]
    n_dies_w = len(y_w)
    defects_w = int(y_w.sum())
    
    # 1. Baseline
    p_b_w = prod_preds[w_mask]
    tp_b = int(((p_b_w == 1) & (y_w == 1)).sum())
    fn_b = int(((p_b_w == 0) & (y_w == 1)).sum())
    fp_b = int(((p_b_w == 1) & (y_w == 0)).sum())
    tn_b = int(((p_b_w == 0) & (y_w == 0)).sum())
    prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
    rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
    f1_b = (2 * prec_b * rec_b) / (prec_b + rec_b) if (prec_b + rec_b) > 0 else 0.0
    
    # 2. EXP-01 Locked
    p_e1_w = exp01_preds[w_mask]
    tp_e1 = int(((p_e1_w == 1) & (y_w == 1)).sum())
    fn_e1 = int(((p_e1_w == 0) & (y_w == 1)).sum())
    fp_e1 = int(((p_e1_w == 1) & (y_w == 0)).sum())
    tn_e1 = int(((p_e1_w == 0) & (y_w == 0)).sum())
    prec_e1 = tp_e1 / (tp_e1 + fp_e1) if (tp_e1 + fp_e1) > 0 else 0.0
    rec_e1 = tp_e1 / (tp_e1 + fn_e1) if (tp_e1 + fn_e1) > 0 else 0.0
    f1_e1 = (2 * prec_e1 * rec_e1) / (prec_e1 + rec_e1) if (prec_e1 + rec_e1) > 0 else 0.0
    
    # 3. EXP-02 Preferred
    p_e2_w = pref_preds[w_mask]
    tp_e2 = int(((p_e2_w == 1) & (y_w == 1)).sum())
    fn_e2 = int(((p_e2_w == 0) & (y_w == 1)).sum())
    fp_e2 = int(((p_e2_w == 1) & (y_w == 0)).sum())
    tn_e2 = int(((p_e2_w == 0) & (y_w == 0)).sum())
    prec_e2 = tp_e2 / (tp_e2 + fp_e2) if (tp_e2 + fp_e2) > 0 else 0.0
    rec_e2 = tp_e2 / (tp_e2 + fn_e2) if (tp_e2 + fn_e2) > 0 else 0.0
    f1_e2 = (2 * prec_e2 * rec_e2) / (prec_e2 + rec_e2) if (prec_e2 + rec_e2) > 0 else 0.0
    
    wafer_records.append({
        'wafer_id': wid,
        'dies_count': n_dies_w,
        'actual_defects': defects_w,
        'baseline_tp': tp_b,
        'baseline_fn': fn_b,
        'baseline_fp': fp_b,
        'baseline_recall': round(rec_b, 4),
        'baseline_precision': round(prec_b, 4),
        'baseline_f1': round(f1_b, 4),
        'exp01_tp': tp_e1,
        'exp01_fn': fn_e1,
        'exp01_fp': fp_e1,
        'exp01_recall': round(rec_e1, 4),
        'exp01_precision': round(prec_e1, 4),
        'exp01_f1': round(f1_e1, 4),
        'exp02_tp': tp_e2,
        'exp02_fn': fn_e2,
        'exp02_fp': fp_e2,
        'exp02_recall': round(rec_e2, 4),
        'exp02_precision': round(prec_e2, 4),
        'exp02_f1': round(f1_e2, 4),
        'delta_f1_vs_baseline': round(f1_e2 - f1_b, 4),
        'delta_f1_vs_exp01': round(f1_e2 - f1_e1, 4),
        'delta_tp_vs_baseline': tp_e2 - tp_b,
        'delta_fp_vs_baseline': fp_e2 - fp_b,
        'status_vs_exp01': 'IMPROVED' if f1_e2 > f1_e1 else ('DEGRADED' if f1_e2 < f1_e1 else 'UNCHANGED'),
        'status_vs_baseline': 'IMPROVED' if f1_e2 > f1_b else ('DEGRADED' if f1_e2 < f1_b else 'UNCHANGED')
    })

wafer_df = pd.DataFrame(wafer_records)
wafer_df.to_csv(os.path.join(EXP_DIR, 'wafer_level_results.csv'), index=False)

n_imp_vs_base = (wafer_df['status_vs_baseline'] == 'IMPROVED').sum()
n_unc_vs_base = (wafer_df['status_vs_baseline'] == 'UNCHANGED').sum()
n_deg_vs_base = (wafer_df['status_vs_baseline'] == 'DEGRADED').sum()

n_imp_vs_e1 = (wafer_df['status_vs_exp01'] == 'IMPROVED').sum()
n_unc_vs_e1 = (wafer_df['status_vs_exp01'] == 'UNCHANGED').sum()
n_deg_vs_e1 = (wafer_df['status_vs_exp01'] == 'DEGRADED').sum()

log(f"  • Wafer F1 vs Baseline: {n_imp_vs_base} improved, {n_unc_vs_base} unchanged, {n_deg_vs_base} degraded")
log(f"  • Wafer F1 vs EXP-01:   {n_imp_vs_e1} improved, {n_unc_vs_e1} unchanged, {n_deg_vs_e1} degraded")
log(f"  • Mean Wafer Recall: Baseline={wafer_df['baseline_recall'].mean():.4f}, EXP-01={wafer_df['exp01_recall'].mean():.4f}, EXP-02={wafer_df['exp02_recall'].mean():.4f}")
log(f"  • Mean Wafer Precision: Baseline={wafer_df['baseline_precision'].mean():.4f}, EXP-01={wafer_df['exp01_precision'].mean():.4f}, EXP-02={wafer_df['exp02_precision'].mean():.4f}")
log(f"  • Mean Wafer F1: Baseline={wafer_df['baseline_f1'].mean():.4f}, EXP-01={wafer_df['exp01_f1'].mean():.4f}, EXP-02={wafer_df['exp02_f1'].mean():.4f}")

# -------------------------------------------------------------
# 7. SPECIAL ANALYSES: TRAPPED FN & PROBABILITY BANDS
# -------------------------------------------------------------
log("\n[6/7] Performing Trapped FN analysis and probability band audit...")

# Section 13 & 32: Trapped FN analysis for gate_lower in [0.15, 0.18, 0.20, 0.22, 0.25, 0.27, 0.30]
# In EXP-01 Model A at 0.5100:
pred_A_051 = (val_probs_exp >= 0.5100).astype(int)
fn_A_051_mask = (pred_A_051 == 0) & (y_val == 1)
total_trapped_fn = int((fn_A_051_mask & (val_probs_exp < 0.25)).sum())
log(f"  • Total trapped FNs in EXP-01 Model A with P_A < 0.25: {total_trapped_fn} / {fn_A_051_mask.sum()}")

gate_special_points = [0.15, 0.18, 0.20, 0.22, 0.25, 0.27, 0.30]
trapped_rows = []
for gl in gate_special_points:
    # dies exposed between gl and 0.25 (newly reachable)
    if gl < 0.25:
        exposed_mask = fn_A_051_mask & (val_probs_exp >= gl) & (val_probs_exp < 0.25)
        new_fp_mask = (y_val == 0) & (val_probs_exp >= gl) & (val_probs_exp < 0.25) & (val_preds_B == 1)
    else:
        exposed_mask = np.zeros(N_VAL, dtype=bool)
        new_fp_mask = np.zeros(N_VAL, dtype=bool)
    
    n_exposed = int(exposed_mask.sum())
    n_rescued = int((exposed_mask & (val_preds_B == 1)).sum())
    rescue_rate = n_rescued / n_exposed if n_exposed > 0 else 0.0
    new_fps = int(new_fp_mask.sum())
    
    # Overall cascade with tau_A=0.5100 and this gate_lower
    esc_mask = (val_probs_exp >= gl) & (val_probs_exp <= LOCKED_GATE_UPPER)
    casc_p = np.where(esc_mask, val_preds_B, pred_A_051)
    tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_val, casc_p).ravel()
    prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
    rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
    f1_c = (2 * prec_c * rec_c) / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0
    f2_c = (5 * prec_c * rec_c) / (4 * prec_c + rec_c) if (4 * prec_c + rec_c) > 0 else 0.0
    denom_mcc = np.sqrt(float(tp_c + fp_c) * (tp_c + fn_c) * (tn_c + fp_c) * (tn_c + fn_c))
    mcc_c = (float(tp_c * tn_c) - float(fp_c * fn_c)) / denom_mcc if denom_mcc > 0 else 0.0
    
    # TPs lost at tau_A=0.5100
    tp_lost = int(((pred_A_051 == 1) & (y_val == 1) & esc_mask & (val_preds_B == 0)).sum())
    net_tp = n_rescued - tp_lost
    
    trapped_rows.append({
        'gate_lower': gl,
        'exposed_original_FNs': n_exposed,
        'rescued_FNs': n_rescued,
        'rescue_rate': round(rescue_rate, 4),
        'new_FPs': new_fps,
        'TPs_lost': tp_lost,
        'net_TP_gain': net_tp,
        'escalation_rate': round(esc_mask.mean(), 4),
        'compute_saved': round(1.0 - esc_mask.mean(), 4),
        'cascade_f1': round(f1_c, 4),
        'cascade_f2': round(f2_c, 4),
        'cascade_recall': round(rec_c, 4),
        'cascade_precision': round(prec_c, 4),
        'cascade_mcc': round(mcc_c, 4),
        'cascade_tp': tp_c,
        'cascade_fn': fn_c,
        'cascade_fp': fp_c
    })

trapped_df = pd.DataFrame(trapped_rows)
trapped_df.to_csv(os.path.join(EXP_DIR, 'trapped_fn_analysis.csv'), index=False)
log("  • Trapped FN analysis table:")
log(trapped_df[['gate_lower', 'exposed_original_FNs', 'rescued_FNs', 'new_FPs', 'escalation_rate', 'cascade_f1', 'cascade_recall', 'cascade_fp']].to_string(index=False))

# Section 12: Recovery Ledger for Preferred, Baseline, and EXP-01
def build_ledger_row(system_name, tau_a, gl, gu):
    pred_a = (val_probs_exp >= tau_a).astype(int) if 'Baseline' not in system_name else (val_probs_base >= tau_a).astype(int)
    p_a = val_probs_exp if 'Baseline' not in system_name else val_probs_base
    esc = (p_a >= gl) & (p_a <= gu)
    casc = np.where(esc, val_preds_B, pred_a)
    
    # FN ledger
    ma_fn = (pred_a == 0) & (y_val == 1)
    ma_fn_sent = ma_fn & esc
    ma_fn_rescued = ma_fn_sent & (val_preds_B == 1)
    casc_fn = (casc == 0) & (y_val == 1)
    
    # TP ledger
    ma_tp = (pred_a == 1) & (y_val == 1)
    ma_tp_sent = ma_tp & esc
    ma_tp_preserved = ma_tp_sent & (val_preds_B == 1)
    ma_tp_lost = ma_tp_sent & (val_preds_B == 0)
    
    # FP ledger
    ma_fp = (pred_a == 1) & (y_val == 0)
    ma_fp_sent = ma_fp & esc
    ma_fp_removed = ma_fp_sent & (val_preds_B == 0)
    ma_fp_remained = ma_fp_sent & (val_preds_B == 1)
    new_fp = (pred_a == 0) & (y_val == 0) & esc & (val_preds_B == 1)
    casc_fp = (casc == 1) & (y_val == 0)
    
    return {
        'system': system_name,
        'tau_A': tau_a,
        'gate_lower': gl,
        'gate_upper': gu,
        'Model_A_FN': int(ma_fn.sum()),
        'FN_sent_to_Model_B': int(ma_fn_sent.sum()),
        'FN_rescued_by_Model_B': int(ma_fn_rescued.sum()),
        'Cascade_remaining_FN': int(casc_fn.sum()),
        'Model_A_TP': int(ma_tp.sum()),
        'TP_sent_to_Model_B': int(ma_tp_sent.sum()),
        'TP_preserved': int(ma_tp_preserved.sum()),
        'TP_lost': int(ma_tp_lost.sum()),
        'Cascade_final_TP': int(((casc == 1) & (y_val == 1)).sum()),
        'Model_A_FP': int(ma_fp.sum()),
        'FP_sent_to_Model_B': int(ma_fp_sent.sum()),
        'FP_removed_by_Model_B': int(ma_fp_removed.sum()),
        'FP_remaining_from_Model_A': int(ma_fp_remained.sum()),
        'New_FP_from_Model_B': int(new_fp.sum()),
        'Cascade_final_FP': int(casc_fp.sum())
    }

ledger_df = pd.DataFrame([
    build_ledger_row('Production_Baseline', 0.5100, 0.25, 0.55),
    build_ledger_row('EXP-01_Locked', 0.5100, 0.25, 0.55),
    build_ledger_row('EXP-02_Preferred', tau_A_pref, gl_pref, gu_pref)
])
ledger_df.to_csv(os.path.join(EXP_DIR, 'recovery_ledger.csv'), index=False)
log("  • Recovery ledger saved to recovery_ledger.csv")

# Section 19: Probability-Band Analysis
bins_standard = [(i/20, (i+1)/20) for i in range(20)]
bins_critical = [(0.15, 0.30), (0.20, 0.35), (0.25, 0.40), (0.40, 0.55), (0.50, 0.60)]

band_rows = []
for low, high in bins_standard + bins_critical:
    mask = (val_probs_exp >= low) & (val_probs_exp < high)
    n_dies = int(mask.sum())
    n_def = int(y_val[mask].sum()) if n_dies > 0 else 0
    emp_rate = n_def / n_dies if n_dies > 0 else 0.0
    mean_p_a = float(val_probs_exp[mask].mean()) if n_dies > 0 else 0.0
    mean_p_b = float(val_probs_B[mask].mean()) if n_dies > 0 else 0.0
    b_pos = int((val_preds_B[mask] == 1).sum()) if n_dies > 0 else 0
    b_prec = int(((val_preds_B[mask] == 1) & (y_val[mask] == 1)).sum()) / b_pos if b_pos > 0 else 0.0
    
    band_type = 'Critical_Transition' if (low, high) in bins_critical else 'Standard_Decile'
    band_rows.append({
        'band_type': band_type,
        'band_interval': f"[{low:.2f}, {high:.2f})",
        'population': n_dies,
        'actual_defects': n_def,
        'empirical_defect_rate': round(emp_rate, 4),
        'mean_prob_Model_A': round(mean_p_a, 4),
        'mean_prob_Model_B': round(mean_p_b, 4),
        'Model_B_positives': b_pos,
        'Model_B_precision_in_band': round(b_prec, 4)
    })

band_df = pd.DataFrame(band_rows)
band_df.to_csv(os.path.join(EXP_DIR, 'probability_band_analysis.csv'), index=False)
log("  • Probability-band analysis saved to probability_band_analysis.csv")

# Section 33: Dedicated tau_A threshold curves
tau_special = [0.40, 0.42, 0.44, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55]
tau_curve_df = grid_df[(grid_df['gate_lower'] == gl_pref) & (grid_df['tau_A'].isin(tau_special))].copy()
tau_curve_df.to_csv(os.path.join(EXP_DIR, 'threshold_curves.csv'), index=False)

# Export Heatmap CSVs
def export_heatmap(metric_name, filename):
    pivot = grid_df.pivot(index='tau_A', columns='gate_lower', values=metric_name)
    pivot.to_csv(os.path.join(EXP_DIR, filename))

export_heatmap('f1', 'heatmap_f1.csv')
export_heatmap('f2', 'heatmap_f2.csv')
export_heatmap('recall', 'heatmap_recall.csv')
export_heatmap('mcc', 'heatmap_mcc.csv')
export_heatmap('fp', 'heatmap_fp.csv')
export_heatmap('fn', 'heatmap_fn.csv')
export_heatmap('escalation_rate', 'heatmap_escalation.csv')
log("  • All 7 heatmap CSVs exported.")

# -------------------------------------------------------------
# 8. PUBLICATION PLOTS GENERATION
# -------------------------------------------------------------
log("\n[7/7] Generating publication-quality diagnostic plots...")

# Plot 1: Pareto Frontier
plt.figure(figsize=(9, 6), dpi=300)
plt.scatter(grid_df['recall'] * 100, grid_df['precision'] * 100, c='lightgray', s=25, alpha=0.5, label='Dominated Grid Points')
plt.scatter(pareto_df['recall'] * 100, pareto_df['precision'] * 100, c='#2563eb', s=45, label='Pareto Frontier')
plt.scatter([cand_max_f1['recall'] * 100], [cand_max_f1['precision'] * 100], c='#16a34a', s=120, marker='*', zorder=5, label=f"EXP-02 Preferred (F1={cand_max_f1['f1']:.4f})")
plt.scatter([cm_prod[3]/N_FAIL * 100], [cm_prod[3]/(cm_prod[3]+cm_prod[1]) * 100], c='#dc2626', s=90, marker='X', zorder=5, label='Production Baseline')
plt.scatter([cm_exp01[3]/N_FAIL * 100], [cm_exp01[3]/(cm_exp01[3]+cm_exp01[1]) * 100], c='#9333ea', s=90, marker='D', zorder=5, label='EXP-01 Locked')
plt.title('EXP-02 Pareto Frontier: Recall vs Precision Operating Points', fontsize=12, fontweight='bold')
plt.xlabel('Cascade Recall (%)', fontsize=11)
plt.ylabel('Cascade Precision (%)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True, loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'pareto_frontier.png'))
plt.close()

# Plot 2: Joint Heatmaps (F1 and F2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
pivot_f1 = grid_df.pivot(index='tau_A', columns='gate_lower', values='f1')
im1 = ax1.imshow(pivot_f1.values, origin='lower', cmap='viridis', aspect='auto',
                 extent=[gate_lower_values[0], gate_lower_values[-1], tau_A_values[0], tau_A_values[-1]])
ax1.set_title('Cascade F1 Score Surface', fontsize=12, fontweight='bold')
ax1.set_xlabel('Gate B Lower Bound', fontsize=11)
ax1.set_ylabel('Model A Threshold (tau_A)', fontsize=11)
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

pivot_f2 = grid_df.pivot(index='tau_A', columns='gate_lower', values='f2')
im2 = ax2.imshow(pivot_f2.values, origin='lower', cmap='plasma', aspect='auto',
                 extent=[gate_lower_values[0], gate_lower_values[-1], tau_A_values[0], tau_A_values[-1]])
ax2.set_title('Cascade F2 Score Surface', fontsize=12, fontweight='bold')
ax2.set_xlabel('Gate B Lower Bound', fontsize=11)
ax2.set_ylabel('Model A Threshold (tau_A)', fontsize=11)
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'heatmap_f1_f2.png'))
plt.close()

# Plot 3: Trapped FN Rescue vs False Alarms
plt.figure(figsize=(9, 5), dpi=300)
plt.plot(trapped_df['gate_lower'], trapped_df['rescued_FNs'], 'o-', color='#16a34a', linewidth=2, label='Rescued FNs (TP Gain)')
plt.plot(trapped_df['gate_lower'], trapped_df['new_FPs'], 's--', color='#dc2626', linewidth=2, label='New False Positives (FP Cost)')
plt.axvline(0.25, color='gray', linestyle=':', label='Baseline Gate (0.25)')
plt.title('Trapped Defect Rescue vs False Alarm Penalty Across Gate B Lower Bound', fontsize=12, fontweight='bold')
plt.xlabel('Gate B Lower Bound', fontsize=11)
plt.ylabel('Die Count', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'trapped_fn_recovery.png'))
plt.close()

# Plot 4: Wafer F1 Comparison Across 32 Wafers
plt.figure(figsize=(14, 5), dpi=300)
x = np.arange(len(wafer_df))
width = 0.28
plt.bar(x - width, wafer_df['baseline_f1'], width, label='Production Baseline', color='#dc2626', alpha=0.8)
plt.bar(x, wafer_df['exp01_f1'], width, label='EXP-01 Locked', color='#9333ea', alpha=0.8)
plt.bar(x + width, wafer_df['exp02_f1'], width, label='EXP-02 Preferred', color='#16a34a', alpha=0.8)
plt.xticks(x, [w.replace('W_N_', '') for w in wafer_df['wafer_id']], rotation=90, fontsize=8)
plt.title('Wafer-Level F1 Robustness Comparison Across All 32 Canonical Validation Wafers', fontsize=12, fontweight='bold')
plt.xlabel('Wafer ID', fontsize=11)
plt.ylabel('F1 Score', fontsize=11)
plt.legend(frameon=True)
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'wafer_robustness.png'))
plt.close()

log("  • All diagnostic plots generated successfully.")

# -------------------------------------------------------------
# 9. POST-EXPERIMENT INTEGRITY & HASH CHECK
# -------------------------------------------------------------
log("\nVerifying post-experiment artifact integrity...")
hashes_after = {}
mutations = []
for rel_path in FILES_TO_LOCK:
    full_p = os.path.join(PROJECT_ROOT, rel_path)
    h = compute_sha256(full_p)
    hashes_after[rel_path] = h
    if h != hashes_before[rel_path]:
        mutations.append(rel_path)

with open(os.path.join(EXP_DIR, 'hashes_after.json'), 'w') as f:
    json.dump(hashes_after, f, indent=2)

if len(mutations) == 0:
    log("✅ POST-EXPERIMENT ZERO MUTATION: All 9 production files remain 100% bit-for-bit intact!")
else:
    log(f"❌ CRITICAL ERROR: Mutated production files detected: {mutations}")
    sys.exit(1)

log("\n================================================================================")
log("EXP-02 EXECUTION COMPLETED SUCCESSFULLY!")
log("================================================================================")
