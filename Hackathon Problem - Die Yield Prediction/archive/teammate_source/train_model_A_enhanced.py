import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, confusion_matrix, brier_score_loss
)
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_wafer_split, load_canonical_wafer_split
from src.evaluation_protocol import (
    validate_eligible_dataset, validate_features_not_leaked,
    validate_wafer_partition_integrity, calculate_evaluation_metrics,
    MODEL_A_LOCKED_THRESHOLD
)

SEED = 42
np.random.seed(SEED)

print("=" * 90, flush=True)
print("EXPERIMENT: ADVANCED MODEL A RECALL & PRECISION OPTIMIZATION (NO BLOCK DATA)", flush=True)
print("=" * 90, flush=True)

# -----------------------------------------------------------------------------
# 1. FAST LOAD DATA (EXCLUDING UNUSED BLOCK STRINGS) & ENFORCE 128/32 SPLIT
# -----------------------------------------------------------------------------
train_path = os.path.join(PROJECT_ROOT, 'input', 'train.csv')
print(f"\n[1/6] Fast-loading tabular parametric data (ignoring block strings) from {train_path}...", flush=True)

# Determine column names without reading large block strings
sample_header = pd.read_csv(train_path, nrows=1)
model_a_cols = [c for c in sample_header.columns if c != 'block_readings']
raw_df = pd.read_csv(train_path, usecols=model_a_cols)

# Filter for eligible dies (old_label == 0)
eligible_df = raw_df[raw_df['old_label'] == 0].reset_index(drop=True)
print(f"  • Loaded {len(eligible_df):,} eligible dies ({len(raw_df):,} total dies in dataset).", flush=True)

train_mask, val_mask, train_wafers, val_wafers = get_wafer_split(eligible_df)
train_df = eligible_df[train_mask].reset_index(drop=True)
val_df = eligible_df[val_mask].reset_index(drop=True)
validate_wafer_partition_integrity(train_df, val_df, train_wafers, val_wafers)

y_train = train_df['label'].values
y_val = val_df['label'].values
feat_cols = [f'feature_{i}' for i in range(1, 501)]

print(f"  • Train: {len(train_wafers)} wafers ({len(train_df):,} dies, {y_train.sum():,} fails, {y_train.mean()*100:.2f}% fail rate)", flush=True)
print(f"  • Val:   {len(val_wafers)} wafers ({len(val_df):,} dies, {y_val.sum():,} fails, {y_val.mean()*100:.2f}% fail rate)", flush=True)

# -----------------------------------------------------------------------------
# 2. ENHANCED SPATIAL FEATURE EXTRACTION
# -----------------------------------------------------------------------------
print("\n[2/6] Computing enhanced polar spatial and multi-scale density features...", flush=True)

def compute_enhanced_spatial_features(df_subset, full_df):
    spatial_list = []
    for wid, group in df_subset.groupby('wafer_id'):
        rows = group['die_row'].values
        cols = group['die_col'].values
        
        r_max = max(rows.max(), 1)
        c_max = max(cols.max(), 1)
        cy, cx = r_max / 2.0, c_max / 2.0
        
        # Radial distance and polar angle
        rad = np.sqrt((rows - cy)**2 + (cols - cx)**2)
        max_r = rad.max() if rad.max() > 0 else 1.0
        norm_rad = rad / max_r
        
        theta = np.arctan2(rows - cy, cols - cx)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Pre-test fail density maps
        full_wafer = full_df[full_df['wafer_id'] == wid]
        grid_rows = full_wafer['die_row'].max() + 1
        grid_cols = full_wafer['die_col'].max() + 1
        
        old_fail_grid = np.zeros((grid_rows, grid_cols), dtype=float)
        valid_grid = np.zeros((grid_rows, grid_cols), dtype=float)
        old_fail_grid[full_wafer['die_row'].values, full_wafer['die_col'].values] = full_wafer['old_label'].values
        valid_grid[full_wafer['die_row'].values, full_wafer['die_col'].values] = 1.0
        
        # Multi-scale density filters
        def get_density(size):
            f_grid = uniform_filter(old_fail_grid, size=size, mode='constant', cval=0.0)
            v_grid = uniform_filter(valid_grid, size=size, mode='constant', cval=0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                d_grid = np.where(v_grid > 0, f_grid / v_grid, 0.0)
            return d_grid[rows, cols]
            
        dens_3 = get_density(3)
        dens_5 = get_density(5)
        dens_7 = get_density(7)
        
        # Edge distance
        edge_dist = np.minimum(np.minimum(rows, r_max - rows), np.minimum(cols, c_max - cols))
        norm_edge_dist = edge_dist / (max(r_max, c_max) / 2.0)
        
        # Wafer pre-test quality prior
        wafer_fail_prior = full_wafer['old_label'].mean()
        
        sub_sp = pd.DataFrame({
            'radial_dist': norm_rad,
            'sin_theta': sin_theta,
            'cos_theta': cos_theta,
            'density_3x3': dens_3,
            'density_5x5': dens_5,
            'density_7x7': dens_7,
            'edge_dist': norm_edge_dist,
            'die_row': rows,
            'die_col': cols,
            'wafer_old_fail_rate': wafer_fail_prior
        }, index=group.index)
        spatial_list.append(sub_sp)
        
    return pd.concat(spatial_list).sort_index()

train_spatial = compute_enhanced_spatial_features(train_df, raw_df)
val_spatial = compute_enhanced_spatial_features(val_df, raw_df)

# -----------------------------------------------------------------------------
# 3. PARAMETRIC STATISTICAL & ANOMALY FEATURE EXTRACTION
# -----------------------------------------------------------------------------
print("\n[3/6] Computing multivariate parametric anomaly & PCA residual features...", flush=True)

X_train_raw = train_df[feat_cols].values
X_val_raw = val_df[feat_cols].values

# Compute pass distribution parameters on train set
pass_mask_tr = (y_train == 0)
pass_mean = X_train_raw[pass_mask_tr].mean(axis=0)
pass_std = np.maximum(X_train_raw[pass_mask_tr].std(axis=0), 1e-6)

# Standardize features relative to train pass distribution
Z_train = (X_train_raw - pass_mean) / pass_std
Z_val = (X_val_raw - pass_mean) / pass_std

# Fit PCA on passing dies
print("  • Fitting PCA (n=25) on clean train passing dies...", flush=True)
pca = PCA(n_components=25, random_state=SEED)
pca.fit(Z_train[pass_mask_tr])

def extract_parametric_summary_features(Z_mat, pca_model):
    abs_z = np.abs(Z_mat)
    max_z = np.max(abs_z, axis=1)
    mean_z = np.mean(abs_z, axis=1)
    std_z = np.std(Z_mat, axis=1)
    sum_sq_z = np.sum(Z_mat**2, axis=1)
    cnt_gt_2 = np.sum(abs_z > 2.0, axis=1)
    cnt_gt_3 = np.sum(abs_z > 3.0, axis=1)
    cnt_gt_4 = np.sum(abs_z > 4.0, axis=1)
    
    # PCA reconstruction residual
    Z_proj = pca_model.transform(Z_mat)
    Z_recon = pca_model.inverse_transform(Z_proj)
    recon_error = np.sum((Z_mat - Z_recon)**2, axis=1)
    
    return pd.DataFrame({
        'param_max_abs_z': max_z,
        'param_mean_abs_z': mean_z,
        'param_std_z': std_z,
        'param_sum_sq_z': sum_sq_z,
        'param_cnt_gt_2sigma': cnt_gt_2,
        'param_cnt_gt_3sigma': cnt_gt_3,
        'param_cnt_gt_4sigma': cnt_gt_4,
        'pca_recon_error': recon_error
    })

train_param_stats = extract_parametric_summary_features(Z_train, pca)
val_param_stats = extract_parametric_summary_features(Z_val, pca)

# Combine full feature matrices
X_train_full = pd.concat([train_df[feat_cols], train_spatial, train_param_stats], axis=1)
X_val_full = pd.concat([val_df[feat_cols], val_spatial, val_param_stats], axis=1)

print(f"  • Total Model A Features: {X_train_full.shape[1]} (500 raw + {train_spatial.shape[1]} spatial + {train_param_stats.shape[1]} parametric stats)", flush=True)

# -----------------------------------------------------------------------------
# 4. TRAIN BASELINE VS ENHANCED MODELS
# -----------------------------------------------------------------------------
print("\n[4/6] Training & comparing model architectures...", flush=True)

results = []

def eval_model(name, probs_val, y_v):
    roc = roc_auc_score(y_v, probs_val)
    p_arr, r_arr, thresh_arr = precision_recall_curve(y_v, probs_val)
    pr_auc = auc(r_arr, p_arr)
    
    # Find optimal F1 threshold
    f1_arr = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(p_arr[:-1], r_arr[:-1])]
    best_f1_idx = np.argmax(f1_arr)
    opt_thresh_f1 = thresh_arr[best_f1_idx]
    
    # Evaluate at optimal F1 threshold
    preds_f1 = (probs_val >= opt_thresh_f1).astype(int)
    f1 = f1_score(y_v, preds_f1)
    prec = precision_score(y_v, preds_f1, zero_division=0)
    rec = recall_score(y_v, preds_f1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_v, preds_f1).ravel()
    brier = brier_score_loss(y_v, probs_val)
    
    return {
        'name': name,
        'f1': f1,
        'auc_pr': pr_auc,
        'roc_auc': roc,
        'precision': prec,
        'recall': rec,
        'opt_thresh': opt_thresh_f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'brier': brier,
        '_probs': probs_val
    }

# A. Baseline LightGBM (500 + 6 spatial)
legacy_spatial_cols = ['radial_dist', 'density_3x3', 'density_5x5', 'edge_dist', 'die_row', 'die_col']
X_tr_base = pd.concat([train_df[feat_cols], train_spatial[legacy_spatial_cols]], axis=1)
X_val_base = pd.concat([val_df[feat_cols], val_spatial[legacy_spatial_cols]], axis=1)

print("  • Training Baseline LightGBM (506 features)...", flush=True)
lgb_base = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    scale_pos_weight=10.0, colsample_bytree=0.8, subsample=0.8,
    random_state=SEED, n_jobs=-1, verbose=-1
)
lgb_base.fit(X_tr_base, y_train)
p_base = lgb_base.predict_proba(X_val_base)[:, 1]
res_base = eval_model("Baseline LightGBM (506 Feats)", p_base, y_val)
results.append(res_base)

# B. Enhanced LightGBM (518 features)
print("  • Training Enhanced LightGBM (518 features)...", flush=True)
lgb_enh = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.03, num_leaves=45,
    scale_pos_weight=10.0, colsample_bytree=0.75, subsample=0.8,
    random_state=SEED, n_jobs=-1, verbose=-1
)
lgb_enh.fit(X_train_full, y_train)
p_enh = lgb_enh.predict_proba(X_val_full)[:, 1]
res_enh = eval_model("Enhanced LightGBM (518 Feats)", p_enh, y_val)
results.append(res_enh)

# C. Check for XGBoost
try:
    import xgboost as xgb
    print("  • Training XGBoost Classifier...", flush=True)
    scale_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=6,
        scale_pos_weight=scale_weight * 0.4, colsample_bytree=0.75,
        subsample=0.8, random_state=SEED, n_jobs=-1, tree_method='hist'
    )
    xgb_clf.fit(X_train_full, y_train)
    p_xgb = xgb_clf.predict_proba(X_val_full)[:, 1]
    res_xgb = eval_model("Enhanced XGBoost (518 Feats)", p_xgb, y_val)
    results.append(res_xgb)
except Exception as e:
    print(f"  [Notice] XGBoost training skipped: {e}", flush=True)
    p_xgb = None

# D. Check for CatBoost
try:
    import catboost as cb
    print("  • Training CatBoost Classifier...", flush=True)
    cb_clf = cb.CatBoostClassifier(
        iterations=600, learning_rate=0.04, depth=6,
        auto_class_weights='Balanced', random_seed=SEED,
        verbose=False, thread_count=-1
    )
    cb_clf.fit(X_train_full, y_train)
    p_cb = cb_clf.predict_proba(X_val_full)[:, 1]
    res_cb = eval_model("Enhanced CatBoost (518 Feats)", p_cb, y_val)
    results.append(res_cb)
except Exception as e:
    print(f"  [Notice] CatBoost training skipped: {e}", flush=True)
    p_cb = None

# E. Blended Tri-GBDT Ensemble
if p_xgb is not None and p_cb is not None:
    ensemble_p = 0.40 * p_enh + 0.35 * p_xgb + 0.25 * p_cb
    res_ens = eval_model("Tri-GBDT Ensemble (LGBM + XGB + CatBoost)", ensemble_p, y_val)
    results.append(res_ens)
elif p_xgb is not None:
    ensemble_p = 0.55 * p_enh + 0.45 * p_xgb
    res_ens = eval_model("Dual-GBDT Ensemble (LGBM + XGBoost)", ensemble_p, y_val)
    results.append(res_ens)

# -----------------------------------------------------------------------------
# 5. SUMMARY COMPARISON TABLE & RECALL SWEEP
# -----------------------------------------------------------------------------
print("\n" + "=" * 90, flush=True)
print(f"{'Model Variant':42s} | {'F1':6s} | {'AUC-PR':6s} | {'Recall':7s} | {'Precision':9s} | {'ROC-AUC':7s} | {'TP':4s} | {'FN':4s} | {'FP':4s}", flush=True)
print("=" * 90, flush=True)
for r in results:
    print(f"{r['name']:42s} | {r['f1']:.4f} | {r['auc_pr']:.4f} | {r['recall']*100:6.2f}% | {r['precision']*100:8.2f}% | {r['roc_auc']:.4f} | {r['tp']:4d} | {r['fn']:4d} | {r['fp']:4d}", flush=True)
print("=" * 90, flush=True)

print("\n--- ENHANCED MODEL A (518 FEATS) THRESHOLD & RECALL SWEEP ---", flush=True)
print(f"{'Operating Threshold':20s} | {'Recall':7s} | {'Precision':9s} | {'F1':6s} | {'TP (Caught)':12s} | {'FN (Missed)':12s} | {'FP (False Alarms)':18s}", flush=True)
print("-" * 95, flush=True)
for t in [0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]:
    preds_t = (p_enh >= t).astype(int)
    rec_t = recall_score(y_val, preds_t, zero_division=0)
    prec_t = precision_score(y_val, preds_t, zero_division=0)
    f1_t = f1_score(y_val, preds_t, zero_division=0)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_val, preds_t).ravel()
    print(f"Threshold = {t:.2f}         | {rec_t*100:6.2f}% | {prec_t*100:8.2f}% | {f1_t:.4f} | {tp_t:12d} | {fn_t:12d} | {fp_t:18d}", flush=True)
print("-" * 95, flush=True)
