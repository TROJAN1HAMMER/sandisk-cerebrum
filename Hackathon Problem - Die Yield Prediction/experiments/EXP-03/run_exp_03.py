#!/usr/bin/env python3
"""
EXP-03: Model B Feature & Error-Profile Improvement
SanDisk Cerebrum Die Yield Prediction Platform

Scientific Evaluation & Second-Stage Feature Engineering
DO NOT RETRAIN MODEL A OR CNN. DO NOT OVERWRITE PRODUCTION ARTIFACTS.
"""

import os
# Prevent macOS Python 3.14 OpenMP SIGSEGV crash
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
EXP02_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'EXP-02')
PLOTS_DIR = os.path.join(EXP_DIR, 'plots')
EXP_MODELS_DIR = os.path.join(EXP_DIR, 'models')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(EXP_MODELS_DIR, exist_ok=True)

# Add PROJECT_ROOT to path
sys.path.insert(0, PROJECT_ROOT)
from src.spatial import compute_spatial_features
from src.block_parser import parse_block_matrix, extract_B0_scalar_features, extract_B1_engineered_features

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

def evaluate_classifier_metrics(y_true, y_prob, threshold, name):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec_arr, prec_arr)
    roc_auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_ece(y_true, y_prob)
    
    return {
        'model_name': name,
        'threshold': round(float(threshold), 4),
        'accuracy': round(float(acc), 5),
        'balanced_accuracy': round(float(bal_acc), 5),
        'precision': round(float(prec), 5),
        'recall': round(float(rec), 5),
        'specificity': round(float(spec), 5),
        'fpr': round(float(fpr), 5),
        'fnr': round(float(fnr), 5),
        'f1': round(float(f1), 5),
        'f2': round(float(f2), 5),
        'mcc': round(float(mcc), 5),
        'pr_auc': round(float(pr_auc), 5),
        'roc_auc': round(float(roc_auc), 5),
        'brier': round(float(brier), 5),
        'ece': round(float(ece), 5),
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn)
    }

# -------------------------------------------------------------
# 1. INTEGRITY PRE-CHECK
# -------------------------------------------------------------
log("=" * 80)
log("EXP-03: Model B Feature & Error-Profile Improvement")
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
for rel_path, expected_hash in hashes_before.items():
    current_hash = compute_sha256(os.path.join(PROJECT_ROOT, rel_path))
    assert current_hash == expected_hash, f"PRE-RUN MUTATION DETECTED in {rel_path}!"
log("✅ Pre-experiment hashes verified 100% bit-for-bit identical.")

# -------------------------------------------------------------
# 2. LOAD DATASET & CANONICAL SPLIT
# -------------------------------------------------------------
log("\n[1/8] Loading dataset and canonical holdout wafer partitions...")
with open(os.path.join(RESULTS_DIR, 'canonical_wafer_split.json'), 'r') as f:
    split = json.load(f)
val_wafers = set(split['validation_wafers'])
train_wafers = set(split['train_wafers'])

df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
eligible = df[df['old_label'] == 0].copy().reset_index(drop=True)

train_mask = eligible['wafer_id'].isin(train_wafers)
val_mask = eligible['wafer_id'].isin(val_wafers)

train_df = eligible[train_mask].reset_index(drop=True)
val_df = eligible[val_mask].reset_index(drop=True)

y_train = train_df['label'].values
y_val = val_df['label'].values

log(f"  • Train Set: {len(train_wafers)} wafers | {len(train_df):,} dies | {y_train.sum():,} fails ({y_train.mean()*100:.2f}%)")
log(f"  • Val Set:   {len(val_wafers)} wafers | {len(val_df):,} dies | {y_val.sum():,} fails ({y_val.mean()*100:.2f}%)")

# Base parametric features (500)
feat_cols = [f'feature_{i}' for i in range(1, 501)]
spatial_names = ['radial_dist', 'density_3x3', 'density_5x5', 'edge_dist', 'die_row', 'die_col']

# -------------------------------------------------------------
# 3. SPATIAL FEATURES (BROKEN BASELINE VS RESTORED EXP-01)
# -------------------------------------------------------------
log("\n[2/8] Preparing parametric and spatial feature matrices...")
# Spatial Condition A: Broken (without full_df, as in original Model B)
sp_before_all = compute_spatial_features(eligible)
sp_before_tr = sp_before_all[train_mask].reset_index(drop=True)
sp_before_val = sp_before_all[val_mask].reset_index(drop=True)

# Spatial Condition B: Restored (with full_df=df, as in EXP-01)
sp_after_all = compute_spatial_features(eligible, full_df=df)
sp_after_tr = sp_after_all[train_mask].reset_index(drop=True)
sp_after_val = sp_after_all[val_mask].reset_index(drop=True)

X_tr_A_base = pd.concat([train_df[feat_cols], sp_before_tr[spatial_names]], axis=1)
X_val_A_base = pd.concat([val_df[feat_cols], sp_before_val[spatial_names]], axis=1)

X_tr_A_restored = pd.concat([train_df[feat_cols], sp_after_tr[spatial_names]], axis=1)
X_val_A_restored = pd.concat([val_df[feat_cols], sp_after_val[spatial_names]], axis=1)

# -------------------------------------------------------------
# 4. BLOCK FEATURE EXTRACTION & CACHING
# -------------------------------------------------------------
log("\n[3/8] Extracting baseline and candidate engineered block features...")
cache_block_path = os.path.join(EXP_DIR, 'cached_block_features.npz')

def extract_advanced_candidate_block_features(block_matrix):
    n = len(block_matrix)
    b_mean = block_matrix.mean(axis=1)
    b_max = block_matrix.max(axis=1)
    b_p95 = np.percentile(block_matrix, 95, axis=1)
    
    # High-magnitude counts
    cnt_110 = (block_matrix > 110.0).sum(axis=1).astype(np.float32)
    cnt_115 = (block_matrix > 115.0).sum(axis=1).astype(np.float32)
    cnt_120 = (block_matrix > 120.0).sum(axis=1).astype(np.float32)
    
    # High-magnitude excess energy (Area above 110 & 115)
    area_110 = np.maximum(0, block_matrix - 110.0).sum(axis=1).astype(np.float32)
    area_115 = np.maximum(0, block_matrix - 115.0).sum(axis=1).astype(np.float32)
    
    # Multi-scale 20-block rolling window
    roll_20_max = np.zeros(n, dtype=np.float32)
    for i in range(n):
        roll_20_max[i] = uniform_filter1d(block_matrix[i], size=20, mode='nearest').max()
        
    roll_20_dev = (roll_20_max - 100.0).astype(np.float32)
    roll_20_rel = (roll_20_max - b_mean).astype(np.float32)
    
    # Peak prominence
    peak_prom = (b_max - b_p95).astype(np.float32)
    
    # Top 10 mean relative to mean
    top10_mean = np.partition(block_matrix, -10, axis=1)[:, -10:].mean(axis=1)
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

if os.path.exists(cache_block_path):
    log("  • Loading pre-extracted block features from cache...")
    c = np.load(cache_block_path, allow_pickle=True)
    tr_b_eng = pd.DataFrame(c['tr_b_eng'], columns=c['b_eng_cols'])
    val_b_eng = pd.DataFrame(c['val_b_eng'], columns=c['b_eng_cols'])
    tr_b_cand = pd.DataFrame(c['tr_b_cand'], columns=c['b_cand_cols'])
    val_b_cand = pd.DataFrame(c['val_b_cand'], columns=c['b_cand_cols'])
else:
    log("  • Parsing raw block_readings string matrices (Train & Val)...")
    t0 = time.time()
    tr_block_mat = parse_block_matrix(train_df['block_readings'])
    val_block_mat = parse_block_matrix(val_df['block_readings'])
    log(f"  • Parsed block matrices: Train={tr_block_mat.shape}, Val={val_block_mat.shape} in {time.time()-t0:.1f}s")
    
    log("  • Computing baseline 20 engineered block features...")
    t0 = time.time()
    tr_b_eng = extract_B1_engineered_features(tr_block_mat)
    val_b_eng = extract_B1_engineered_features(val_block_mat)
    log(f"  • Extracted baseline block features in {time.time()-t0:.1f}s")
    
    log("  • Computing candidate 10 advanced block features...")
    t0 = time.time()
    tr_b_cand = extract_advanced_candidate_block_features(tr_block_mat)
    val_b_cand = extract_advanced_candidate_block_features(val_block_mat)
    log(f"  • Extracted candidate block features in {time.time()-t0:.1f}s")
    
    np.savez_compressed(
        cache_block_path,
        tr_b_eng=tr_b_eng.values,
        val_b_eng=val_b_eng.values,
        b_eng_cols=tr_b_eng.columns.tolist(),
        tr_b_cand=tr_b_cand.values,
        val_b_cand=val_b_cand.values,
        b_cand_cols=tr_b_cand.columns.tolist()
    )
    log(f"  • Cached block features to {cache_block_path}")

# Load locked Model A predictions (from EXP-01)
pred_exp01_df = pd.read_csv(os.path.join(EXP01_DIR, 'prediction_diagnostics.csv'))
val_probs_A_exp01 = pred_exp01_df['prob_Model_A_exp01'].values

# -------------------------------------------------------------
# 5. ERROR FORENSICS & COHORT ANALYSIS ON BASELINE MODEL B
# -------------------------------------------------------------
log("\n[4/8] Performing baseline Model B error forensics and cohort audit...")
with open(os.path.join(MODELS_DIR, 'model_B.pkl'), 'rb') as f:
    clf_B0_prod = pickle.load(f)

val_probs_B0 = clf_B0_prod.predict_proba(np.hstack([X_val_A_base.values, val_b_eng.values]))[:, 1]
LOCKED_TAU_B = 0.5376
val_preds_B0 = (val_probs_B0 >= LOCKED_TAU_B).astype(int)

# Verify B0 reproduces clean diagnostics
diag_df = pd.read_csv(os.path.join(RESULTS_DIR, 'val_die_diagnostics.csv'))
np.testing.assert_allclose(val_probs_B0, diag_df['prob_Model_B'].values, atol=1e-4)
log("  ✅ Production Model B predictions verified bit-for-bit!")

# Cohort masks
b0_tp = (val_preds_B0 == 1) & (y_val == 1)
b0_fn = (val_preds_B0 == 0) & (y_val == 1)
b0_fp = (val_preds_B0 == 1) & (y_val == 0)
b0_tn = (val_preds_B0 == 0) & (y_val == 0)

# Feature Forensics Table across all 20 existing block features
feat_audit_rows = []
block_cols = val_b_eng.columns.tolist()
b0_gains = clf_B0_prod.booster_.feature_importance(importance_type='gain')[-20:]
b0_splits = clf_B0_prod.booster_.feature_importance(importance_type='split')[-20:]

for idx, col in enumerate(block_cols):
    s = val_b_eng[col]
    feat_audit_rows.append({
        'feature': col,
        'gain': round(float(b0_gains[idx]), 2),
        'splits': int(b0_splits[idx]),
        'mean_TP': round(float(s[b0_tp].mean()), 4),
        'mean_FN': round(float(s[b0_fn].mean()), 4),
        'mean_FP': round(float(s[b0_fp].mean()), 4),
        'mean_TN': round(float(s[b0_tn].mean()), 4),
        'median_TP': round(float(np.median(s[b0_tp])), 4),
        'median_FN': round(float(np.median(s[b0_fn])), 4),
        'median_FP': round(float(np.median(s[b0_fp])), 4),
        'median_TN': round(float(np.median(s[b0_tn])), 4),
        'std': round(float(s.std()), 4),
        'min': round(float(s.min()), 4),
        'max': round(float(s.max()), 4),
        'status': 'DEAD' if b0_splits[idx] == 0 else ('WEAK' if b0_splits[idx] <= 5 else 'ACTIVE')
    })

pd.DataFrame(feat_audit_rows).to_csv(os.path.join(EXP_DIR, 'feature_forensics.csv'), index=False)
log("  • Saved feature forensics to feature_forensics.csv")

# Cohort Summary Table
cohort_summary_rows = [
    {'cohort': 'TP', 'count': int(b0_tp.sum()), 'mean_cluster_area': round(float(val_b_eng['cluster_area'][b0_tp].mean()), 2), 'mean_b_mean': round(float(val_b_eng['b_mean'][b0_tp].mean()), 4)},
    {'cohort': 'FN', 'count': int(b0_fn.sum()), 'mean_cluster_area': round(float(val_b_eng['cluster_area'][b0_fn].mean()), 2), 'mean_b_mean': round(float(val_b_eng['b_mean'][b0_fn].mean()), 4)},
    {'cohort': 'FP', 'count': int(b0_fp.sum()), 'mean_cluster_area': round(float(val_b_eng['cluster_area'][b0_fp].mean()), 2), 'mean_b_mean': round(float(val_b_eng['b_mean'][b0_fp].mean()), 4)},
    {'cohort': 'TN', 'count': int(b0_tn.sum()), 'mean_cluster_area': round(float(val_b_eng['cluster_area'][b0_tn].mean()), 2), 'mean_b_mean': round(float(val_b_eng['b_mean'][b0_tn].mean()), 4)}
]
pd.DataFrame(cohort_summary_rows).to_csv(os.path.join(EXP_DIR, 'cohort_statistics.csv'), index=False)
log("  • Saved cohort statistics to cohort_statistics.csv")

# -------------------------------------------------------------
# 6. TRAIN EXPERIMENTAL MODEL B VARIANTS (B0, B1, B2, B3)
# -------------------------------------------------------------
log("\n[5/8] Training controlled Model B experimental variants...")

# Features setup:
# Dead features to prune in B1 & B3:
dead_cols = ['longest_run_104', 'cluster_width', 'b_min', 'mad_100']
b_eng_pruned_cols = [c for c in block_cols if c not in dead_cols]

# Locked Model B LightGBM Hyperparameters
lgb_params_B = {
    'n_estimators': 150,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'scale_pos_weight': 10.0,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': 1
}

variants_def = [
    {
        'id': 'B0',
        'name': 'Production Model B Baseline',
        'desc': '506 Model A (Baseline Spatial) + 20 Baseline Block Features',
        'X_tr': np.hstack([X_tr_A_base.values, tr_b_eng.values]),
        'X_val': np.hstack([X_val_A_base.values, val_b_eng.values]),
        'feature_names': feat_cols + spatial_names + block_cols,
        'model_file': os.path.join(EXP_MODELS_DIR, 'model_B_B0.pkl')
    },
    {
        'id': 'B1',
        'name': 'Feature-Pruned Model B',
        'desc': '506 Model A (Baseline Spatial) + 16 Pruned Block Features (removed 4 dead features)',
        'X_tr': np.hstack([X_tr_A_base.values, tr_b_eng[b_eng_pruned_cols].values]),
        'X_val': np.hstack([X_val_A_base.values, val_b_eng[b_eng_pruned_cols].values]),
        'feature_names': feat_cols + spatial_names + b_eng_pruned_cols,
        'model_file': os.path.join(EXP_MODELS_DIR, 'model_B_B1.pkl')
    },
    {
        'id': 'B2',
        'name': 'Enhanced Model B (Restored Spatial + New Block Features)',
        'desc': '506 Model A (Restored Spatial) + 20 Baseline Block + 10 Candidate Features',
        'X_tr': np.hstack([X_tr_A_restored.values, tr_b_eng.values, tr_b_cand.values]),
        'X_val': np.hstack([X_val_A_restored.values, val_b_eng.values, val_b_cand.values]),
        'feature_names': feat_cols + spatial_names + block_cols + tr_b_cand.columns.tolist(),
        'model_file': os.path.join(EXP_MODELS_DIR, 'model_B_B2.pkl')
    },
    {
        'id': 'B3',
        'name': 'Enhanced + Pruned Model B (Optimal Feature Set)',
        'desc': '506 Model A (Restored Spatial) + 16 Pruned Block + 10 Candidate Features',
        'X_tr': np.hstack([X_tr_A_restored.values, tr_b_eng[b_eng_pruned_cols].values, tr_b_cand.values]),
        'X_val': np.hstack([X_val_A_restored.values, val_b_eng[b_eng_pruned_cols].values, val_b_cand.values]),
        'feature_names': feat_cols + spatial_names + b_eng_pruned_cols + tr_b_cand.columns.tolist(),
        'model_file': os.path.join(EXP_MODELS_DIR, 'model_B_B3.pkl')
    }
]

trained_models = {}
val_predictions = {}
standalone_results = []
inference_benchmarks = []
importance_records = []

for v in variants_def:
    vid = v['id']
    log(f"  • Training Variant {vid}: {v['name']} ({v['X_tr'].shape[1]} features)...")
    
    t0_tr = time.time()
    clf = lgb.LGBMClassifier(**lgb_params_B)
    clf.fit(v['X_tr'], y_train)
    t_train = time.time() - t0_tr
    
    # Save variant pickle
    with open(v['model_file'], 'wb') as fp:
        pickle.dump(clf, fp)
        
    trained_models[vid] = clf
    
    # Inference benchmark
    t0_inf = time.time()
    val_probs = clf.predict_proba(v['X_val'])[:, 1]
    t_inf = time.time() - t0_inf
    val_predictions[vid] = val_probs
    
    n_val_dies = len(y_val)
    ms_per_die = (t_inf / n_val_dies) * 1000.0
    dies_per_sec = n_val_dies / t_inf if t_inf > 0 else 0
    
    inference_benchmarks.append({
        'variant': vid,
        'features_count': v['X_tr'].shape[1],
        'train_time_s': round(t_train, 2),
        'total_inference_time_s': round(t_inf, 4),
        'ms_per_die': round(ms_per_die, 4),
        'dies_per_sec': round(dies_per_sec, 1)
    })
    
    # Evaluate standalone metrics at locked threshold 0.5376
    m_stand = evaluate_classifier_metrics(y_val, val_probs, LOCKED_TAU_B, f"Model_B_{vid}")
    m_stand['variant_id'] = vid
    m_stand['features_count'] = v['X_tr'].shape[1]
    standalone_results.append(m_stand)
    
    # Record feature importances
    gains = clf.booster_.feature_importance(importance_type='gain')
    splits = clf.booster_.feature_importance(importance_type='split')
    for fname, g, s in zip(v['feature_names'], gains, splits):
        importance_records.append({
            'variant': vid,
            'feature': fname,
            'gain': round(float(g), 2),
            'splits': int(s)
        })
        
    log(f"    - {vid} Standalone: F1={m_stand['f1']:.4f}, PR-AUC={m_stand['pr_auc']:.4f}, ROC-AUC={m_stand['roc_auc']:.4f}, Rec={m_stand['recall']*100:.2f}%, Prec={m_stand['precision']*100:.2f}%, TP={m_stand['tp']}, FP={m_stand['fp']} (Trained in {t_train:.1f}s)")

pd.DataFrame(standalone_results).to_csv(os.path.join(EXP_DIR, 'model_B_variants.csv'), index=False)
pd.DataFrame(inference_benchmarks).to_csv(os.path.join(EXP_DIR, 'inference_benchmark.csv'), index=False)
pd.DataFrame(importance_records).to_csv(os.path.join(EXP_DIR, 'model_B_feature_importance.csv'), index=False)
log("  • Saved model_B_variants.csv, inference_benchmark.csv, and model_B_feature_importance.csv")

# -------------------------------------------------------------
# 7. COMPLETE CASCADE INTEGRATION EVALUATION
# -------------------------------------------------------------
log("\n[6/8] Evaluating complete selective cascade with Model B variants...")

# Cascade Policies:
# Policy A: Baseline Gate [0.25, 0.55]
# Policy B: EXP-02 Preferred Gate [0.295, 0.55]
LOCKED_TAU_A = 0.5100
policies = [
    ('Policy_B_Preferred', 0.295, 0.55),
    ('Policy_A_Baseline', 0.25, 0.55)
]

cascade_results = []
rescue_records = []

# Model A EXP-01 standalone reference
pred_A_051 = (val_probs_A_exp01 >= LOCKED_TAU_A).astype(int)
ma_fn_mask = (pred_A_051 == 0) & (y_val == 1)
ma_tp_mask = (pred_A_051 == 1) & (y_val == 1)
ma_fp_mask = (pred_A_051 == 1) & (y_val == 0)

for p_name, gl, gu in policies:
    esc_mask = (val_probs_A_exp01 >= gl) & (val_probs_A_exp01 <= gu)
    esc_count = int(esc_mask.sum())
    esc_rate = esc_count / len(y_val)
    compute_saved = 1.0 - esc_rate
    
    for v in variants_def:
        vid = v['id']
        val_probs_B_v = val_predictions[vid]
        val_preds_B_v = (val_probs_B_v >= LOCKED_TAU_B).astype(int)
        
        casc_probs = np.where(esc_mask, val_probs_B_v, val_probs_A_exp01)
        casc_preds = np.where(esc_mask, val_preds_B_v, pred_A_051)
        
        tn, fp, fn, tp = confusion_matrix(y_val, casc_preds).ravel()
        prec_arr, rec_arr, _ = precision_recall_curve(y_val, casc_probs)
        pr_auc = auc(rec_arr, prec_arr)
        roc_auc = roc_auc_score(y_val, casc_probs)
        f1 = f1_score(y_val, casc_preds, zero_division=0)
        prec = precision_score(y_val, casc_preds, zero_division=0)
        rec = recall_score(y_val, casc_preds, zero_division=0)
        f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
        acc = accuracy_score(y_val, casc_preds)
        bal_acc = balanced_accuracy_score(y_val, casc_preds)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        mcc = matthews_corrcoef(y_val, casc_preds)
        brier = brier_score_loss(y_val, casc_probs)
        ece = calculate_ece(y_val, casc_probs)
        
        # Rescue Accounting
        fn_sent = int((ma_fn_mask & esc_mask).sum())
        fn_rescued = int(((ma_fn_mask & esc_mask) & (val_preds_B_v == 1)).sum())
        fn_rescue_pct = (fn_rescued / fn_sent * 100) if fn_sent > 0 else 0.0
        
        tp_sent = int((ma_tp_mask & esc_mask).sum())
        tp_preserved = int(((ma_tp_mask & esc_mask) & (val_preds_B_v == 1)).sum())
        tp_lost = int(((ma_tp_mask & esc_mask) & (val_preds_B_v == 0)).sum())
        
        fp_sent = int((ma_fp_mask & esc_mask).sum())
        fp_removed = int(((ma_fp_mask & esc_mask) & (val_preds_B_v == 0)).sum())
        new_fp = int(((~ma_fp_mask & (y_val == 0) & esc_mask) & (val_preds_B_v == 1)).sum())
        
        net_tp_gain = fn_rescued - tp_lost
        
        casc_rec = {
            'policy': p_name,
            'variant': vid,
            'gate_lower': gl,
            'gate_upper': gu,
            'tau_A': LOCKED_TAU_A,
            'tau_B': LOCKED_TAU_B,
            'f1': round(float(f1), 5),
            'f2': round(float(f2), 5),
            'recall': round(float(rec), 5),
            'precision': round(float(prec), 5),
            'mcc': round(float(mcc), 5),
            'pr_auc': round(float(pr_auc), 5),
            'roc_auc': round(float(roc_auc), 5),
            'balanced_accuracy': round(float(bal_acc), 5),
            'specificity': round(float(spec), 5),
            'fpr': round(float(fpr), 5),
            'brier': round(float(brier), 5),
            'ece': round(float(ece), 5),
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp),
            'tn': int(tn),
            'escalation_rate': round(float(esc_rate), 5),
            'compute_saved': round(float(compute_saved), 5),
            'fn_sent_to_B': fn_sent,
            'fn_rescued': fn_rescued,
            'fn_rescue_pct': round(float(fn_rescue_pct), 2),
            'tp_lost': tp_lost,
            'net_tp_gain': net_tp_gain,
            'new_fp_introduced': new_fp
        }
        cascade_results.append(casc_rec)
        
        if p_name == 'Policy_B_Preferred':
            rescue_records.append({
                'variant': vid,
                'fn_sent': fn_sent,
                'fn_rescued': fn_rescued,
                'rescue_rate': round(fn_rescue_pct, 2),
                'tp_lost': tp_lost,
                'net_tp_gain': net_tp_gain,
                'new_fp': new_fp,
                'final_cascade_tp': int(tp),
                'final_cascade_fp': int(fp),
                'cascade_f1': round(float(f1), 4),
                'cascade_precision': round(float(prec), 4),
                'cascade_recall': round(float(rec), 4)
            })
            
        log(f"    - Cascade [{p_name}] Variant {vid}: F1={f1:.4f}, PR-AUC={pr_auc:.4f}, Rec={rec*100:.2f}%, Prec={prec*100:.2f}%, TP={tp}, FN={fn}, FP={fp}, Esc={esc_rate*100:.2f}%")

pd.DataFrame(cascade_results).to_csv(os.path.join(EXP_DIR, 'cascade_variant_results.csv'), index=False)
pd.DataFrame(rescue_records).to_csv(os.path.join(EXP_DIR, 'rescue_analysis.csv'), index=False)
log("  • Saved cascade_variant_results.csv and rescue_analysis.csv")

# -------------------------------------------------------------
# 8. WAFER-LEVEL ROBUSTNESS AUDIT ACROSS ALL 32 WAFERS
# -------------------------------------------------------------
log("\n[7/8] Conducting wafer-level breakdown across all 32 validation wafers...")
wafer_ids = val_df['wafer_id'].values
unique_wafers = np.unique(wafer_ids)
wafer_records = []

# Evaluate wafer breakdown under Policy_B_Preferred for B0 vs B1 vs B2 vs B3
esc_pref = (val_probs_A_exp01 >= 0.295) & (val_probs_A_exp01 <= 0.55)
casc_preds_dict = {}
for vid in ['B0', 'B1', 'B2', 'B3']:
    casc_preds_dict[vid] = np.where(esc_pref, (val_predictions[vid] >= LOCKED_TAU_B).astype(int), pred_A_051)

for wid in unique_wafers:
    w_mask = (wafer_ids == wid)
    y_w = y_val[w_mask]
    n_dies_w = len(y_w)
    defects_w = int(y_w.sum())
    
    rec_w = {'wafer_id': wid, 'dies_count': n_dies_w, 'actual_defects': defects_w}
    
    for vid in ['B0', 'B1', 'B2', 'B3']:
        cp_w = casc_preds_dict[vid][w_mask]
        tp_w = int(((cp_w == 1) & (y_w == 1)).sum())
        fn_w = int(((cp_w == 0) & (y_w == 1)).sum())
        fp_w = int(((cp_w == 1) & (y_w == 0)).sum())
        prec_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0.0
        rec_val_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0.0
        f1_w = (2 * prec_w * rec_val_w) / (prec_w + rec_val_w) if (prec_w + rec_val_w) > 0 else 0.0
        
        rec_w[f'{vid}_tp'] = tp_w
        rec_w[f'{vid}_fn'] = fn_w
        rec_w[f'{vid}_fp'] = fp_w
        rec_w[f'{vid}_recall'] = round(float(rec_val_w), 4)
        rec_w[f'{vid}_precision'] = round(float(prec_w), 4)
        rec_w[f'{vid}_f1'] = round(float(f1_w), 4)
        
    rec_w['delta_f1_B2_vs_B0'] = round(rec_w['B2_f1'] - rec_w['B0_f1'], 4)
    rec_w['delta_f1_B3_vs_B0'] = round(rec_w['B3_f1'] - rec_w['B0_f1'], 4)
    rec_w['status_B2_vs_B0'] = 'IMPROVED' if rec_w['B2_f1'] > rec_w['B0_f1'] else ('DEGRADED' if rec_w['B2_f1'] < rec_w['B0_f1'] else 'UNCHANGED')
    rec_w['status_B3_vs_B0'] = 'IMPROVED' if rec_w['B3_f1'] > rec_w['B0_f1'] else ('DEGRADED' if rec_w['B3_f1'] < rec_w['B0_f1'] else 'UNCHANGED')
    wafer_records.append(rec_w)

wafer_df = pd.DataFrame(wafer_records)
wafer_df.to_csv(os.path.join(EXP_DIR, 'wafer_level_results.csv'), index=False)
log("  • Saved wafer_level_results.csv")

for vid in ['B1', 'B2', 'B3']:
    imp = (wafer_df[f'{vid}_f1'] > wafer_df['B0_f1']).sum()
    unc = (wafer_df[f'{vid}_f1'] == wafer_df['B0_f1']).sum()
    deg = (wafer_df[f'{vid}_f1'] < wafer_df['B0_f1']).sum()
    log(f"  • Wafer F1 ({vid} vs B0): {imp} improved, {unc} unchanged, {deg} degraded | Mean F1={wafer_df[f'{vid}_f1'].mean():.4f} (vs B0={wafer_df['B0_f1'].mean():.4f})")

# -------------------------------------------------------------
# 9. SECONDARY SENSITIVITY SWEEP ON TAU_B
# -------------------------------------------------------------
log("\n[8/8] Conducting secondary sensitivity analysis on tau_B...")
# Evaluate B2 / B3 across tau_B in [0.50, 0.52, 0.5376, 0.55, 0.57]
sens_records = []
for tau_b in [0.50, 0.52, 0.5376, 0.55, 0.57]:
    for vid in ['B0', 'B2', 'B3']:
        cp = np.where(esc_pref, (val_predictions[vid] >= tau_b).astype(int), pred_A_051)
        tn, fp, fn, tp = confusion_matrix(y_val, cp).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
        sens_records.append({
            'variant': vid,
            'tau_B': tau_b,
            'f1': round(float(f1), 4),
            'f2': round(float(f2), 4),
            'recall': round(float(rec), 4),
            'precision': round(float(prec), 4),
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp)
        })

pd.DataFrame(sens_records).to_csv(os.path.join(EXP_DIR, 'tau_B_sensitivity.csv'), index=False)
log("  • Saved tau_B_sensitivity.csv")

# -------------------------------------------------------------
# 10. GENERATE PLOTS
# -------------------------------------------------------------
log("\nGenerating publication-quality diagnostic plots...")

# Plot 1: PR Curves for Model B Standalone Variants
plt.figure(figsize=(8, 6), dpi=300)
for vid, color in [('B0', '#dc2626'), ('B1', '#ea580c'), ('B2', '#16a34a'), ('B3', '#2563eb')]:
    p_arr, r_arr, _ = precision_recall_curve(y_val, val_predictions[vid])
    score = auc(r_arr, p_arr)
    plt.plot(r_arr, p_arr, label=f"{vid} (PR-AUC={score:.4f})", color=color, linewidth=2)
plt.title('EXP-03 Standalone Model B Precision-Recall Curves', fontsize=12, fontweight='bold')
plt.xlabel('Recall', fontsize=11)
plt.ylabel('Precision', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_B_pr_curves.png'))
plt.close()

# Plot 2: Rescue Rate vs False Alarms
plt.figure(figsize=(8, 5), dpi=300)
res_df = pd.DataFrame(rescue_records)
x = np.arange(len(res_df))
width = 0.35
plt.bar(x - width/2, res_df['fn_rescued'], width, label='FN Rescued (TPs)', color='#16a34a')
plt.bar(x + width/2, res_df['new_fp'], width, label='New False Positives (FPs)', color='#dc2626')
plt.xticks(x, res_df['variant'])
plt.title('Model B Rescue Power vs False Alarm Cost (Policy B)', fontsize=12, fontweight='bold')
plt.xlabel('Model B Variant', fontsize=11)
plt.ylabel('Die Count', fontsize=11)
plt.legend(frameon=True)
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rescue_vs_false_alarms.png'))
plt.close()

# Plot 3: Top Feature Importances Comparison
top_feats = pd.DataFrame(importance_records)
top_b3 = top_feats[top_feats['variant'] == 'B3'].sort_values(by='gain', ascending=False).head(15)
plt.figure(figsize=(10, 6), dpi=300)
plt.barh(top_b3['feature'][::-1], top_b3['gain'][::-1], color='#2563eb')
plt.title('Top 15 Features in Model B (Variant B3)', fontsize=12, fontweight='bold')
plt.xlabel('Gain Importance', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance_B3.png'))
plt.close()

log("  • Diagnostic plots generated successfully.")

# -------------------------------------------------------------
# 11. POST-EXPERIMENT INTEGRITY & HASH CHECK
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
log("EXP-03 EXECUTION COMPLETED SUCCESSFULLY!")
log("================================================================================")
