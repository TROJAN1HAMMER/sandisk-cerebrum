#!/usr/bin/env python3
"""
EXP-04: Independent Holdout Generalization & Final Model Validation
SanDisk Cerebrum Die Yield Prediction Platform

Strict Independent Validation of Frozen B2 Candidate Cascade on input/test.csv.
NO HYPERPARAMETER TUNING. NO THRESHOLD SWEEPS. NO MODEL FITTING.
"""

import os
# Prevent macOS Python 3.14 OpenMP crash
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
from scipy.ndimage import uniform_filter1d

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
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXP_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
INPUT_DIR = os.path.join(PROJECT_ROOT, 'input')
PLOTS_DIR = os.path.join(EXP_DIR, 'plots')
EXP03B_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'EXP-03B')
EXP01_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'EXP-01')

os.makedirs(PLOTS_DIR, exist_ok=True)

# Add PROJECT_ROOT to path
sys.path.insert(0, PROJECT_ROOT)
from src.spatial import compute_spatial_features
from src.block_parser import parse_block_matrix, extract_B1_engineered_features

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
log("EXP-04: Independent Holdout Generalization & Final Model Validation")
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
with open(hashes_before_path, 'r') as f:
    hashes_before = json.load(f)

log("Verifying pre-experiment file hashes...")
for rel_path in FILES_TO_LOCK:
    full_p = os.path.join(PROJECT_ROOT, rel_path)
    cur_h = compute_sha256(full_p)
    assert cur_h == hashes_before[rel_path], f"Pre-hash violation for {rel_path}!"
log("✅ Pre-experiment hashes verified 100% bit-for-bit identical.")

# -------------------------------------------------------------
# 2. LOAD INDEPENDENT EVALUATION DATASET
# -------------------------------------------------------------
log("\n[1/7] Loading independent ground-truth evaluation population (input/test.csv)...")
t0 = time.time()
test_csv_path = os.path.join(INPUT_DIR, 'test.csv')
assert os.path.exists(test_csv_path), "Missing input/test.csv!"

df_raw_test = pd.read_csv(test_csv_path)
total_raw_rows = len(df_raw_test)
raw_wafers = df_raw_test['wafer_id'].nunique()

log(f"  • Total raw rows loaded: {total_raw_rows:,} across {raw_wafers} unseen wafers")

# Pre-test eligibility filter: old_label == 0 (functional pass at wafer probe)
eligible_mask = (df_raw_test['old_label'] == 0)
pretest_rejects = int((~eligible_mask).sum())
eligible_df = df_raw_test[eligible_mask].copy().reset_index(drop=True)
n_eligible = len(eligible_df)
actual_defects = int(eligible_df['label'].sum())
defect_rate = actual_defects / n_eligible * 100.0
clean_yield = 100.0 - defect_rate

log(f"  • Pre-test rejects (old_label == 1): {pretest_rejects:,} ({pretest_rejects/total_raw_rows*100:.2f}%)")
log(f"  • Eligible dies evaluated: {n_eligible:,} ({n_eligible/total_raw_rows*100:.2f}%)")
log(f"  • Actual latent defects (label == 1): {actual_defects:,} ({defect_rate:.4f}%)")
log(f"  • Ground-truth die yield: {clean_yield:.4f}%")

# Population audit
has_missing = eligible_df.isnull().any().any()
has_duplicates = eligible_df.duplicated(subset=['wafer_id', 'die_row', 'die_col']).any()
log(f"  • Missing values: {has_missing} | Duplicates: {has_duplicates}")

# -------------------------------------------------------------
# 3. SPATIAL & BLOCK FEATURE EXTRACTION
# -------------------------------------------------------------
log("\n[2/7] Extracting spatial context and frozen engineered block features...")

# Spatial features using full wafer map (old_label from raw dataset)
sp_test = compute_spatial_features(eligible_df, full_df=df_raw_test)
spatial_cols = ['radial_dist', 'density_3x3', 'density_5x5', 'edge_dist', 'die_row', 'die_col']
feat_cols = [f'feature_{i}' for i in range(1, 501)]

X_test_A = pd.concat([eligible_df[feat_cols], sp_test[spatial_cols]], axis=1)
log(f"  • Model A feature matrix prepared: {X_test_A.shape}")

# Block features extraction
t_block = time.time()
block_matrix = parse_block_matrix(eligible_df['block_readings'])
b1_feats = extract_B1_engineered_features(block_matrix)

# Frozen 10 candidate block features from EXP-03 / EXP-03B
def extract_advanced_candidate_block_features(block_mat):
    n = len(block_mat)
    b_mean = block_mat.mean(axis=1)
    b_max = block_mat.max(axis=1)
    b_p95 = np.percentile(block_mat, 95, axis=1)
    
    cnt_110 = (block_mat > 110.0).sum(axis=1).astype(np.float32)
    cnt_115 = (block_mat > 115.0).sum(axis=1).astype(np.float32)
    cnt_120 = (block_mat > 120.0).sum(axis=1).astype(np.float32)
    
    area_110 = np.maximum(0, block_mat - 110.0).sum(axis=1).astype(np.float32)
    area_115 = np.maximum(0, block_mat - 115.0).sum(axis=1).astype(np.float32)
    
    roll_20_max = np.zeros(n, dtype=np.float32)
    for i in range(n):
        roll_20_max[i] = uniform_filter1d(block_mat[i], size=20, mode='nearest').max()
        
    roll_20_dev = (roll_20_max - 100.0).astype(np.float32)
    roll_20_rel = (roll_20_max - b_mean).astype(np.float32)
    peak_prom = (b_max - b_p95).astype(np.float32)
    
    top10_mean = np.partition(block_mat, -10, axis=1)[:, -10:].mean(axis=1)
    top10_dev = (top10_mean - b_mean).astype(np.float32)
    
    cols = [
        'cnt_110', 'cnt_115', 'cnt_120', 'cluster_area_110', 'cluster_area_115',
        'max_roll20', 'max_roll20_dev', 'max_roll20_rel', 'peak_prominence', 'top10_dev'
    ]
    feats = np.column_stack([
        cnt_110, cnt_115, cnt_120, area_110, area_115,
        roll_20_max, roll_20_dev, roll_20_rel, peak_prom, top10_dev
    ])
    return pd.DataFrame(feats, columns=cols)

b_cand_feats = extract_advanced_candidate_block_features(block_matrix)
X_test_B2 = np.hstack([X_test_A.values, b1_feats.values, b_cand_feats.values])
log(f"  • Block features extracted in {time.time() - t_block:.2f}s | B2 feature matrix: {X_test_B2.shape}")

# -------------------------------------------------------------
# 4. LOAD FROZEN MODEL CHECKPOINTS
# -------------------------------------------------------------
log("\n[3/7] Loading frozen Model A and confirmed Model B B2 checkpoints...")

model_A_path = os.path.join(EXP01_DIR, 'model_A_spatial.pkl')
with open(model_A_path, 'rb') as f:
    clf_A = pickle.load(f)

model_B2_path = os.path.join(EXP03B_DIR, 'models', 'model_B_B2_confirmed.pkl')
with open(model_B2_path, 'rb') as f:
    clf_B2 = pickle.load(f)

log("  ✅ Frozen Model A and B2 checkpoints loaded successfully.")

# -------------------------------------------------------------
# 5. INFERENCE & CASCADE EXECUTION
# -------------------------------------------------------------
log("\n[4/7] Executing frozen cascade inference (Policy B: Gate [0.295, 0.550], tau_A=0.5100, tau_B=0.5376)...")

LOCKED_TAU_A = 0.5100
LOCKED_TAU_B = 0.5376
GATE_LOWER = 0.295
GATE_UPPER = 0.550

t_inf_start = time.perf_counter()
p_A = clf_A.predict_proba(X_test_A)[:, 1]
t_inf_A = time.perf_counter()

p_B = clf_B2.predict_proba(X_test_B2)[:, 1]
t_inf_B = time.perf_counter()

pred_A = (p_A >= LOCKED_TAU_A).astype(int)
pred_B = (p_B >= LOCKED_TAU_B).astype(int)

# Gating decisions
esc_mask = (p_A >= GATE_LOWER) & (p_A <= GATE_UPPER)
esc_count = int(esc_mask.sum())
esc_rate = esc_count / n_eligible
compute_saved = 1.0 - esc_rate

# Complete cascade predictions
casc_probs = np.where(esc_mask, p_B, p_A)
casc_preds = np.where(esc_mask, pred_B, pred_A)

# Timing metrics
ms_per_die_A = ((t_inf_A - t_inf_start) / n_eligible) * 1000.0
ms_per_die_B = ((t_inf_B - t_inf_A) / n_eligible) * 1000.0
effective_ms_per_die = ms_per_die_A + (esc_rate * ms_per_die_B)
effective_throughput = 1000.0 / effective_ms_per_die

log(f"  • Model A inference latency: {ms_per_die_A:.4f} ms/die")
log(f"  • Model B inference latency: {ms_per_die_B:.4f} ms/die")
log(f"  • Cascade effective latency: {effective_ms_per_die:.4f} ms/die ({effective_throughput:,.0f} dies/sec)")
log(f"  • Escalated dies: {esc_count:,} / {n_eligible:,} ({esc_rate*100:.2f}%) | Compute saved: {compute_saved*100:.2f}%")

# Save per-die predictions
y_true = eligible_df['label'].values

error_types = []
for yt, yp in zip(y_true, casc_preds):
    if yt == 1 and yp == 1:
        error_types.append('TP')
    elif yt == 0 and yp == 1:
        error_types.append('FP')
    elif yt == 1 and yp == 0:
        error_types.append('FN')
    else:
        error_types.append('TN')

gate_decisions = np.where(p_A > GATE_UPPER, 'FAST_FAIL',
                          np.where(p_A < GATE_LOWER, 'FAST_PASS', 'ESCALATED'))

df_preds = pd.DataFrame({
    'wafer_id': eligible_df['wafer_id'],
    'row': eligible_df['die_row'],
    'col': eligible_df['die_col'],
    'actual_label': y_true,
    'p_A': np.round(p_A, 5),
    'gate_decision': gate_decisions,
    'escalated': esc_mask.astype(int),
    'p_B': np.round(p_B, 5),
    'final_prediction': casc_preds,
    'error_type': error_types
})
df_preds.to_csv(os.path.join(EXP_DIR, 'independent_predictions.csv'), index=False)
log("  ✅ Saved independent_predictions.csv successfully.")

# -------------------------------------------------------------
# 6. METRIC COMPUTATION & RESCUE FORENSICS
# -------------------------------------------------------------
log("\n[5/7] Computing primary metrics, Model A baselines, conditional quality, and rescue mechanics...")

# 1. Cascade metrics
tn, fp, fn, tp = confusion_matrix(y_true, casc_preds).ravel()
prec_arr, rec_arr, _ = precision_recall_curve(y_true, casc_probs)
pr_auc = auc(rec_arr, prec_arr)
roc_auc = roc_auc_score(y_true, casc_probs)
f1 = f1_score(y_true, casc_preds, zero_division=0)
prec = precision_score(y_true, casc_preds, zero_division=0)
rec = recall_score(y_true, casc_preds, zero_division=0)
f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
acc = accuracy_score(y_true, casc_preds)
bal_acc = balanced_accuracy_score(y_true, casc_preds)
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
mcc = matthews_corrcoef(y_true, casc_preds)
brier = brier_score_loss(y_true, casc_probs)
ece = calculate_ece(y_true, casc_probs)

# 2. Model A standalone baseline metrics on independent holdout
tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_true, pred_A).ravel()
prec_arr_a, rec_arr_a, _ = precision_recall_curve(y_true, p_A)
pr_auc_a = auc(rec_arr_a, prec_arr_a)
roc_auc_a = roc_auc_score(y_true, p_A)
f1_a = f1_score(y_true, pred_A, zero_division=0)
prec_a = precision_score(y_true, pred_A, zero_division=0)
rec_a = recall_score(y_true, pred_A, zero_division=0)
f2_a = (5 * prec_a * rec_a) / (4 * prec_a + rec_a) if (4 * prec_a + rec_a) > 0 else 0.0
mcc_a = matthews_corrcoef(y_true, pred_A)

# 3. Model B standalone metrics on independent holdout (unconditional)
tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true, pred_B).ravel()
prec_arr_b, rec_arr_b, _ = precision_recall_curve(y_true, p_B)
pr_auc_b = auc(rec_arr_b, prec_arr_b)
roc_auc_b = roc_auc_score(y_true, p_B)
f1_b = f1_score(y_true, pred_B, zero_division=0)
prec_b = precision_score(y_true, pred_B, zero_division=0)
rec_b = recall_score(y_true, pred_B, zero_division=0)
f2_b = (5 * prec_b * rec_b) / (4 * prec_b + rec_b) if (4 * prec_b + rec_b) > 0 else 0.0
mcc_b = matthews_corrcoef(y_true, pred_B)

# 4. Model B conditional quality on escalated cohort
y_cond = y_true[esc_mask]
p_b_cond = p_B[esc_mask]
pred_b_cond = pred_B[esc_mask]
tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_cond, pred_b_cond).ravel()
prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
f1_c = (2 * prec_c * rec_c) / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0
f2_c = (5 * prec_c * rec_c) / (4 * prec_c + rec_c) if (4 * prec_c + rec_c) > 0 else 0.0
mcc_c = matthews_corrcoef(y_cond, pred_b_cond)
prec_arr_c, rec_arr_c, _ = precision_recall_curve(y_cond, p_b_cond)
pr_auc_c = auc(rec_arr_c, prec_arr_c)
roc_auc_c = roc_auc_score(y_cond, p_b_cond)
brier_c = brier_score_loss(y_cond, p_b_cond)

# 5. Rescue Mechanics
ma_fn_mask = (y_true == 1) & (pred_A == 0)
ma_tp_mask = (y_true == 1) & (pred_A == 1)
ma_fp_mask = (y_true == 0) & (pred_A == 1)

fn_sent = int((ma_fn_mask & esc_mask).sum())
fn_rescued = int(((ma_fn_mask & esc_mask) & (pred_B == 1)).sum())
fn_missed = fn_sent - fn_rescued
rescue_rate = (fn_rescued / fn_sent * 100.0) if fn_sent > 0 else 0.0

tp_sent = int((ma_tp_mask & esc_mask).sum())
tp_lost = int(((ma_tp_mask & esc_mask) & (pred_B == 0)).sum())
tp_preserved = tp_sent - tp_lost

fp_sent = int((ma_fp_mask & esc_mask).sum())
fp_removed = int(((ma_fp_mask & esc_mask) & (pred_B == 0)).sum())
new_fp = int(((~ma_fp_mask & (y_true == 0) & esc_mask) & (pred_B == 1)).sum())
net_tp = fn_rescued - tp_lost

log(f"\n--- Cascade Performance Summary ---")
log(f"  • Cascade F1: {f1:.4f} | F2: {f2:.4f} | MCC: {mcc:.4f} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")
log(f"  • Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}")
log(f"  • Model A Baseline: F1={f1_a:.4f}, Prec={prec_a*100:.2f}%, Rec={rec_a*100:.2f}%, TP={tp_a}, FP={fp_a}")
log(f"  • Rescue: {fn_rescued}/{fn_sent} FNs rescued ({rescue_rate:.2f}%) | {tp_lost} TPs lost | Net TP gain: +{net_tp} | New FP: +{new_fp}")
log(f"  • Conditional Model B (Escalated cohort N={esc_count}): F1={f1_c:.4f}, Prec={prec_c*100:.2f}%, Rec={rec_c*100:.2f}%, PR-AUC={pr_auc_c:.4f}")

# Save metrics JSON
metrics_dict = {
    'dataset': 'input/test.csv',
    'total_dies': int(n_eligible),
    'actual_defects': int(actual_defects),
    'defect_rate_pct': round(float(defect_rate), 4),
    'escalation_rate_pct': round(float(esc_rate * 100.0), 4),
    'compute_saved_pct': round(float(compute_saved * 100.0), 4),
    'cascade': {
        'f1': round(float(f1), 5),
        'f2': round(float(f2), 5),
        'precision': round(float(prec), 5),
        'recall': round(float(rec), 5),
        'mcc': round(float(mcc), 5),
        'pr_auc': round(float(pr_auc), 5),
        'roc_auc': round(float(roc_auc), 5),
        'accuracy': round(float(acc), 5),
        'balanced_accuracy': round(float(bal_acc), 5),
        'specificity': round(float(spec), 5),
        'fpr': round(float(fpr), 5),
        'fnr': round(float(fnr), 5),
        'brier': round(float(brier), 5),
        'ece': round(float(ece), 5),
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn)
    },
    'model_A_baseline': {
        'f1': round(float(f1_a), 5),
        'f2': round(float(f2_a), 5),
        'precision': round(float(prec_a), 5),
        'recall': round(float(rec_a), 5),
        'mcc': round(float(mcc_a), 5),
        'pr_auc': round(float(pr_auc_a), 5),
        'roc_auc': round(float(roc_auc_a), 5),
        'tp': int(tp_a),
        'fn': int(fn_a),
        'fp': int(fp_a),
        'tn': int(tn_a)
    },
    'model_B_standalone': {
        'f1': round(float(f1_b), 5),
        'f2': round(float(f2_b), 5),
        'precision': round(float(prec_b), 5),
        'recall': round(float(rec_b), 5),
        'mcc': round(float(mcc_b), 5),
        'pr_auc': round(float(pr_auc_b), 5),
        'roc_auc': round(float(roc_auc_b), 5),
        'tp': int(tp_b),
        'fn': int(fn_b),
        'fp': int(fp_b),
        'tn': int(tn_b)
    },
    'conditional_model_B': {
        'escalated_dies': int(esc_count),
        'defects_in_cohort': int(y_cond.sum()),
        'f1': round(float(f1_c), 5),
        'f2': round(float(f2_c), 5),
        'precision': round(float(prec_c), 5),
        'recall': round(float(rec_c), 5),
        'mcc': round(float(mcc_c), 5),
        'pr_auc': round(float(pr_auc_c), 5),
        'roc_auc': round(float(roc_auc_c), 5),
        'brier': round(float(brier_c), 5),
        'tp': int(tp_c),
        'fn': int(fn_c),
        'fp': int(fp_c),
        'tn': int(tn_c)
    },
    'rescue_mechanics': {
        'model_A_fn_total': int(fn_a),
        'fn_escalated': int(fn_sent),
        'fn_rescued': int(fn_rescued),
        'fn_missed': int(fn_missed),
        'rescue_rate_pct': round(float(rescue_rate), 2),
        'model_A_tp_escalated': int(tp_sent),
        'tp_lost': int(tp_lost),
        'tp_preserved': int(tp_preserved),
        'net_tp_gain': int(net_tp),
        'new_fp_introduced': int(new_fp)
    }
}

with open(os.path.join(EXP_DIR, 'independent_metrics.json'), 'w') as f:
    json.dump(metrics_dict, f, indent=2)

# Confusion matrix CSV
df_cm = pd.DataFrame([
    {'Predicted': 'Negative (0)', 'Actual Negative (0)': tn, 'Actual Positive (1)': fn},
    {'Predicted': 'Positive (1)', 'Actual Negative (0)': fp, 'Actual Positive (1)': tp}
])
df_cm.to_csv(os.path.join(EXP_DIR, 'independent_confusion_matrix.csv'), index=False)

# -------------------------------------------------------------
# 7. WAFER-LEVEL ANALYSIS & STATISTICAL BOOTSTRAP
# -------------------------------------------------------------
log("\n[6/7] Computing per-wafer metrics and executing paired wafer bootstrap...")

wafer_ids = eligible_df['wafer_id'].values
unique_wafers = np.unique(wafer_ids)
wafer_records = []

for wid in unique_wafers:
    w_mask = (wafer_ids == wid)
    y_w = y_true[w_mask]
    preds_w = casc_preds[w_mask]
    n_w = len(y_w)
    def_w = int(y_w.sum())
    
    tp_w = int(((preds_w == 1) & (y_w == 1)).sum())
    fn_w = int(((preds_w == 0) & (y_w == 1)).sum())
    fp_w = int(((preds_w == 1) & (y_w == 0)).sum())
    tn_w = int(((preds_w == 0) & (y_w == 0)).sum())
    
    prec_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0.0
    rec_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0.0
    f1_w = (2 * prec_w * rec_w) / (prec_w + rec_w) if (prec_w + rec_w) > 0 else 0.0
    f2_w = (5 * prec_w * rec_w) / (4 * prec_w + rec_w) if (4 * prec_w + rec_w) > 0 else 0.0
    mcc_w = matthews_corrcoef(y_w, preds_w) if (len(np.unique(y_w)) > 1 and len(np.unique(preds_w)) > 1) else 0.0
    
    wafer_records.append({
        'wafer_id': wid,
        'population': n_w,
        'actual_defects': def_w,
        'yield_pct': round((1.0 - def_w / n_w) * 100.0, 2),
        'tp': tp_w,
        'fn': fn_w,
        'fp': fp_w,
        'tn': tn_w,
        'precision': round(float(prec_w), 4),
        'recall': round(float(rec_w), 4),
        'f1': round(float(f1_w), 4),
        'f2': round(float(f2_w), 4),
        'mcc': round(float(mcc_w), 4)
    })

df_wafer = pd.DataFrame(wafer_records)
df_wafer.to_csv(os.path.join(EXP_DIR, 'wafer_level_metrics.csv'), index=False)

mean_f1_w = df_wafer['f1'].mean()
median_f1_w = df_wafer['f1'].median()
std_f1_w = df_wafer['f1'].std()
min_f1_w = df_wafer[df_wafer['actual_defects'] > 0]['f1'].min()
worst_w = df_wafer[df_wafer['actual_defects'] > 0].sort_values(by='f1').iloc[0]

log(f"  • Wafer F1: Mean={mean_f1_w:.4f}, Median={median_f1_w:.4f}, Std={std_f1_w:.4f}, Worst={worst_w['wafer_id']} ({worst_w['f1']:.4f})")

# Paired wafer bootstrap (1000 resamples over 40 wafers)
log("  • Executing paired wafer bootstrap (1,000 iterations, seed=42)...")
rng = np.random.default_rng(42)
B_ITERS = 1000
boot_f1 = []
boot_f2 = []
boot_mcc = []
boot_pr_auc = []

wafer_indices = {wid: np.where(wafer_ids == wid)[0] for wid in unique_wafers}

for b in range(B_ITERS):
    sampled_w = rng.choice(unique_wafers, size=len(unique_wafers), replace=True)
    sampled_idxs = np.concatenate([wafer_indices[w] for w in sampled_w])
    
    y_b = y_true[sampled_idxs]
    pred_b = casc_preds[sampled_idxs]
    prob_b = casc_probs[sampled_idxs]
    
    f1_b_val = f1_score(y_b, pred_b, zero_division=0)
    prec_b_val = precision_score(y_b, pred_b, zero_division=0)
    rec_b_val = recall_score(y_b, pred_b, zero_division=0)
    f2_b_val = (5 * prec_b_val * rec_b_val) / (4 * prec_b_val + rec_b_val) if (4 * prec_b_val + rec_b_val) > 0 else 0.0
    mcc_b_val = matthews_corrcoef(y_b, pred_b)
    
    p_arr, r_arr, _ = precision_recall_curve(y_b, prob_b)
    pr_auc_b_val = auc(r_arr, p_arr)
    
    boot_f1.append(f1_b_val)
    boot_f2.append(f2_b_val)
    boot_mcc.append(mcc_b_val)
    boot_pr_auc.append(pr_auc_b_val)

ci_f1 = (np.percentile(boot_f1, 2.5), np.percentile(boot_f1, 97.5))
ci_f2 = (np.percentile(boot_f2, 2.5), np.percentile(boot_f2, 97.5))
ci_mcc = (np.percentile(boot_mcc, 2.5), np.percentile(boot_mcc, 97.5))
ci_pr_auc = (np.percentile(boot_pr_auc, 2.5), np.percentile(boot_pr_auc, 97.5))

bootstrap_results = {
    'iterations': B_ITERS,
    'seed': 42,
    'f1': {'point_estimate': round(float(f1), 4), 'ci_95_lower': round(float(ci_f1[0]), 4), 'ci_95_upper': round(float(ci_f1[1]), 4)},
    'f2': {'point_estimate': round(float(f2), 4), 'ci_95_lower': round(float(ci_f2[0]), 4), 'ci_95_upper': round(float(ci_f2[1]), 4)},
    'mcc': {'point_estimate': round(float(mcc), 4), 'ci_95_lower': round(float(ci_mcc[0]), 4), 'ci_95_upper': round(float(ci_mcc[1]), 4)},
    'pr_auc': {'point_estimate': round(float(pr_auc), 4), 'ci_95_lower': round(float(ci_pr_auc[0]), 4), 'ci_95_upper': round(float(ci_pr_auc[1]), 4)}
}

with open(os.path.join(EXP_DIR, 'bootstrap_results.json'), 'w') as f:
    json.dump(bootstrap_results, f, indent=2)

log(f"  • Bootstrap 95% CIs: F1=[{ci_f1[0]:.4f}, {ci_f1[1]:.4f}], PR-AUC=[{ci_pr_auc[0]:.4f}, {ci_pr_auc[1]:.4f}], MCC=[{ci_mcc[0]:.4f}, {ci_mcc[1]:.4f}]")

# -------------------------------------------------------------
# 8. GENERALIZATION & COMPARISON TABLES
# -------------------------------------------------------------
log("\n[7/7] Generating generalization comparison and publication diagnostic plots...")

# Generalization comparison (EXP-03B vs EXP-04)
# Values from confirmed EXP-03B under Policy B Preferred:
exp03b_ref = {
    'PR_AUC': 0.47768,
    'ROC_AUC': 0.79857,
    'precision': 0.89318,
    'recall': 0.37216,
    'F1': 0.52540,
    'F2': 0.42131,
    'MCC': 0.56739,
    'brier': 0.04170,
    'ece': 0.12316,
    'TP': 393,
    'FN': 663,
    'FP': 47,
    'TN': 26869,
    'escalation': 0.05977,
    'compute_saved': 0.94023
}

gen_rows = [
    {'metric': 'PR-AUC', 'EXP_03B': exp03b_ref['PR_AUC'], 'EXP_04': pr_auc},
    {'metric': 'ROC-AUC', 'EXP_03B': exp03b_ref['ROC_AUC'], 'EXP_04': roc_auc},
    {'metric': 'Precision', 'EXP_03B': exp03b_ref['precision'], 'EXP_04': prec},
    {'metric': 'Recall', 'EXP_03B': exp03b_ref['recall'], 'EXP_04': rec},
    {'metric': 'F1 Score', 'EXP_03B': exp03b_ref['F1'], 'EXP_04': f1},
    {'metric': 'F2 Score', 'EXP_03B': exp03b_ref['F2'], 'EXP_04': f2},
    {'metric': 'MCC', 'EXP_03B': exp03b_ref['MCC'], 'EXP_04': mcc},
    {'metric': 'Brier Score', 'EXP_03B': exp03b_ref['brier'], 'EXP_04': brier},
    {'metric': 'ECE', 'EXP_03B': exp03b_ref['ece'], 'EXP_04': ece},
    {'metric': 'TP', 'EXP_03B': exp03b_ref['TP'], 'EXP_04': tp},
    {'metric': 'FN', 'EXP_03B': exp03b_ref['FN'], 'EXP_04': fn},
    {'metric': 'FP', 'EXP_03B': exp03b_ref['FP'], 'EXP_04': fp},
    {'metric': 'TN', 'EXP_03B': exp03b_ref['TN'], 'EXP_04': tn},
    {'metric': 'Escalation Rate', 'EXP_03B': exp03b_ref['escalation'], 'EXP_04': esc_rate},
    {'metric': 'Compute Saved', 'EXP_03B': exp03b_ref['compute_saved'], 'EXP_04': compute_saved}
]

for r in gen_rows:
    abs_d = r['EXP_04'] - r['EXP_03B']
    rel_d = (abs_d / r['EXP_03B'] * 100.0) if r['EXP_03B'] != 0 else 0.0
    r['abs_delta'] = round(float(abs_d), 5)
    r['rel_delta_pct'] = round(float(rel_d), 2)
    r['EXP_03B'] = round(float(r['EXP_03B']), 5) if isinstance(r['EXP_03B'], float) else r['EXP_03B']
    r['EXP_04'] = round(float(r['EXP_04']), 5) if isinstance(r['EXP_04'], float) else r['EXP_04']

df_gen = pd.DataFrame(gen_rows)
df_gen.to_csv(os.path.join(EXP_DIR, 'generalization_comparison.csv'), index=False)

# Final model comparison (Production Baseline vs EXP-03B B2 vs EXP-04 B2)
final_comp = [
    {
        'model_pipeline': 'Production Baseline Cascade (EXP-01)',
        'eval_population': 'Canonical Holdout (32 Wafers)',
        'f1': 0.5072,
        'f2': 0.4000,
        'mcc': 0.5400,
        'precision': 0.8138,
        'recall': 0.3684,
        'pr_auc': 0.4677,
        'roc_auc': 0.7990,
        'fp': 89,
        'tp': 389,
        'compute_saved': 0.8814
    },
    {
        'model_pipeline': 'Confirmed B2 Cascade (EXP-03B)',
        'eval_population': 'Canonical Holdout (32 Wafers)',
        'f1': round(exp03b_ref['F1'], 4),
        'f2': round(exp03b_ref['F2'], 4),
        'mcc': round(exp03b_ref['MCC'], 4),
        'precision': round(exp03b_ref['precision'], 4),
        'recall': round(exp03b_ref['recall'], 4),
        'pr_auc': round(exp03b_ref['PR_AUC'], 4),
        'roc_auc': round(exp03b_ref['ROC_AUC'], 4),
        'fp': exp03b_ref['FP'],
        'tp': exp03b_ref['TP'],
        'compute_saved': round(exp03b_ref['compute_saved'], 4)
    },
    {
        'model_pipeline': 'Final Validated B2 Cascade (EXP-04)',
        'eval_population': 'Independent Ground-Truth (40 Wafers)',
        'f1': round(float(f1), 4),
        'f2': round(float(f2), 4),
        'mcc': round(float(mcc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'pr_auc': round(float(pr_auc), 4),
        'roc_auc': round(float(roc_auc), 4),
        'fp': int(fp),
        'tp': int(tp),
        'compute_saved': round(float(compute_saved), 4)
    }
]
pd.DataFrame(final_comp).to_csv(os.path.join(EXP_DIR, 'final_model_comparison.csv'), index=False)

# Plots
# 1. Confusion Matrix Plot
plt.figure(figsize=(6, 5), dpi=300)
cm = np.array([[tn, fp], [fn, tp]])
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Independent Holdout Confusion Matrix (EXP-04 B2)', fontweight='bold')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Pass (0)', 'Fail (1)'])
plt.yticks(tick_marks, ['Pass (0)', 'Fail (1)'])
thresh_cm = cm.max() / 2.
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i, j]:,}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh_cm else "black",
                 fontsize=12, fontweight='bold')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
plt.close()

# 2. Reliability Curve Plot
plt.figure(figsize=(6, 5), dpi=300)
prob_true, prob_pred = calibration_curve(y_true, casc_probs, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color='#2563eb', label=f'B2 Cascade (ECE={ece:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='#94a3b8', label='Perfect Calibration')
plt.title('Reliability Calibration Curve (Independent Holdout)', fontweight='bold')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives (Empirical Yield Loss)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'reliability_curve.png'))
plt.close()

# 3. Probability Distribution Plot
plt.figure(figsize=(7, 5), dpi=300)
plt.hist(casc_probs[y_true == 0], bins=50, alpha=0.6, color='#16a34a', label='Passing Dies (Label=0)', log=True)
plt.hist(casc_probs[y_true == 1], bins=50, alpha=0.6, color='#dc2626', label='Defect Dies (Label=1)', log=True)
plt.axvline(LOCKED_TAU_A, color='#d97706', linestyle='--', linewidth=1.5, label=f'Model A Tau ({LOCKED_TAU_A})')
plt.axvline(LOCKED_TAU_B, color='#9333ea', linestyle='--', linewidth=1.5, label=f'Model B Tau ({LOCKED_TAU_B})')
plt.title('Cascade Probability Distribution by True Class (Log Scale)', fontweight='bold')
plt.xlabel('Cascade Predicted Probability')
plt.ylabel('Die Count (Log Scale)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'probability_distribution.png'))
plt.close()

# 4. Wafer Performance Distribution Plot
plt.figure(figsize=(8, 5), dpi=300)
try:
    plt.boxplot(df_wafer['f1'], tick_labels=['Independent Wafers (N=40)'], patch_artist=True,
                boxprops=dict(facecolor='#dbeafe', color='#1d4ed8'),
                medianprops=dict(color='#dc2626', linewidth=2))
except TypeError:
    plt.boxplot(df_wafer['f1'], labels=['Independent Wafers (N=40)'], patch_artist=True,
                boxprops=dict(facecolor='#dbeafe', color='#1d4ed8'),
                medianprops=dict(color='#dc2626', linewidth=2))
plt.title('Wafer-Level F1 Distribution Across 40 Independent Test Wafers', fontweight='bold')
plt.ylabel('Wafer F1 Score')
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'wafer_performance.png'))
plt.close()

log("  ✅ Publication plots generated successfully.")

# -------------------------------------------------------------
# 9. POST-EXPERIMENT ZERO-MUTATION HASH CHECK
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
log("EXP-04 EXECUTION COMPLETED SUCCESSFULLY!")
log("================================================================================")
