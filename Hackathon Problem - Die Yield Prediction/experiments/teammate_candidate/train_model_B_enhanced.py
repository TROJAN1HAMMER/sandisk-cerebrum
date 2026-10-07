import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.ndimage import uniform_filter
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, confusion_matrix, brier_score_loss
)
import lightgbm as lgb

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_wafer_split
from src.evaluation_protocol import validate_wafer_partition_integrity
from src.model_b import Block1DCNN, _get_model_outputs_in_batches

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 90, flush=True)
print("EXPERIMENT: ADVANCED MODEL B & DUAL-STREAM LATE FUSION OPTIMIZATION", flush=True)
print("=" * 90, flush=True)

# -----------------------------------------------------------------------------
# 1. LOAD COMPLETE DATASET & ENFORCE 128 / 32 SPLIT
# -----------------------------------------------------------------------------
train_path = os.path.join(PROJECT_ROOT, 'input', 'train.csv')
print(f"\n[1/7] Loading full dataset (parametric + block readings) from {train_path}...", flush=True)
t0 = time.time()
raw_df = pd.read_csv(train_path)
eligible_df = raw_df[raw_df['old_label'] == 0].reset_index(drop=True)
print(f"  • Loaded {len(eligible_df):,} eligible dies in {time.time() - t0:.1f}s.", flush=True)

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
# 2. COMPUTE MODEL A ENHANCED TABULAR & SPATIAL FEATURES (518 FEATURES)
# -----------------------------------------------------------------------------
print("\n[2/7] Computing Enhanced Model A spatial & parametric anomaly features...", flush=True)

def compute_enhanced_spatial_features(df_subset, full_df):
    spatial_list = []
    for wid, group in df_subset.groupby('wafer_id'):
        rows = group['die_row'].values
        cols = group['die_col'].values
        r_max = max(rows.max(), 1)
        c_max = max(cols.max(), 1)
        cy, cx = r_max / 2.0, c_max / 2.0
        
        rad = np.sqrt((rows - cy)**2 + (cols - cx)**2)
        norm_rad = rad / (rad.max() if rad.max() > 0 else 1.0)
        theta = np.arctan2(rows - cy, cols - cx)
        
        full_wafer = full_df[full_df['wafer_id'] == wid]
        grid_rows = full_wafer['die_row'].max() + 1
        grid_cols = full_wafer['die_col'].max() + 1
        
        old_fail_grid = np.zeros((grid_rows, grid_cols), dtype=float)
        valid_grid = np.zeros((grid_rows, grid_cols), dtype=float)
        old_fail_grid[full_wafer['die_row'].values, full_wafer['die_col'].values] = full_wafer['old_label'].values
        valid_grid[full_wafer['die_row'].values, full_wafer['die_col'].values] = 1.0
        
        def get_density(size):
            f_grid = uniform_filter(old_fail_grid, size=size, mode='constant', cval=0.0)
            v_grid = uniform_filter(valid_grid, size=size, mode='constant', cval=0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                d_grid = np.where(v_grid > 0, f_grid / v_grid, 0.0)
            return d_grid[rows, cols]
            
        edge_dist = np.minimum(np.minimum(rows, r_max - rows), np.minimum(cols, c_max - cols)) / (max(r_max, c_max) / 2.0)
        
        sub_sp = pd.DataFrame({
            'radial_dist': norm_rad,
            'sin_theta': np.sin(theta),
            'cos_theta': np.cos(theta),
            'density_3x3': get_density(3),
            'density_5x5': get_density(5),
            'density_7x7': get_density(7),
            'edge_dist': edge_dist,
            'die_row': rows,
            'die_col': cols,
            'wafer_old_fail_rate': full_wafer['old_label'].mean()
        }, index=group.index)
        spatial_list.append(sub_sp)
        
    return pd.concat(spatial_list).sort_index()

train_spatial = compute_enhanced_spatial_features(train_df, raw_df)
val_spatial = compute_enhanced_spatial_features(val_df, raw_df)

# Parametric stats & PCA
pass_mean = train_df[feat_cols].values[y_train == 0].mean(axis=0)
pass_std = np.maximum(train_df[feat_cols].values[y_train == 0].std(axis=0), 1e-6)
Z_train = (train_df[feat_cols].values - pass_mean) / pass_std
Z_val = (val_df[feat_cols].values - pass_mean) / pass_std

pca = PCA(n_components=25, random_state=SEED)
pca.fit(Z_train[y_train == 0])

def extract_param_stats(Z_mat, pca_model):
    abs_z = np.abs(Z_mat)
    Z_proj = pca_model.transform(Z_mat)
    Z_recon = pca_model.inverse_transform(Z_proj)
    return pd.DataFrame({
        'param_max_abs_z': np.max(abs_z, axis=1),
        'param_mean_abs_z': np.mean(abs_z, axis=1),
        'param_std_z': np.std(Z_mat, axis=1),
        'param_sum_sq_z': np.sum(Z_mat**2, axis=1),
        'param_cnt_gt_2sigma': np.sum(abs_z > 2.0, axis=1),
        'param_cnt_gt_3sigma': np.sum(abs_z > 3.0, axis=1),
        'pca_recon_error': np.sum((Z_mat - Z_recon)**2, axis=1)
    })

train_param_stats = extract_param_stats(Z_train, pca)
val_param_stats = extract_param_stats(Z_val, pca)

X_train_A = pd.concat([train_df[feat_cols], train_spatial, train_param_stats], axis=1)
X_val_A = pd.concat([val_df[feat_cols], val_spatial, val_param_stats], axis=1)

# -----------------------------------------------------------------------------
# 3. PARSE 2,000 BLOCK READINGS & EXTRACT PHYSICS ANOMALY FEATURES
# -----------------------------------------------------------------------------
print("\n[3/7] Parsing 2,000 block sequences & extracting peak burst features...", flush=True)

def parse_blocks_fast(series_blocks):
    print("  • Converting space-delimited block strings to numpy matrix...", flush=True)
    n = len(series_blocks)
    mat = np.zeros((n, 2000), dtype=np.float32)
    for i, s in enumerate(series_blocks):
        mat[i] = np.fromstring(s, sep=' ', dtype=np.float32)
    return mat

train_blocks = parse_blocks_fast(train_df['block_readings'])
val_blocks = parse_blocks_fast(val_df['block_readings'])

def compute_block_burst_features(block_mat):
    print("  • Computing burst streak and peak anomaly features...", flush=True)
    n = len(block_mat)
    
    # 1. Basic descriptive statistics
    b_mean = np.mean(block_mat, axis=1)
    b_std = np.std(block_mat, axis=1)
    b_min = np.min(block_mat, axis=1)
    b_max = np.max(block_mat, axis=1)
    b_p95 = np.percentile(block_mat, 95, axis=1)
    b_p99 = np.percentile(block_mat, 99, axis=1)
    
    # 2. Peak anomaly excursions (threshold 104.0 and 106.0)
    cnt_104 = np.sum(block_mat > 104.0, axis=1)
    cnt_106 = np.sum(block_mat > 106.0, axis=1)
    max_spike_dev = np.max(np.abs(block_mat - 100.0), axis=1)
    
    # 3. Signal volatility (first difference jumps)
    first_diff = np.mean(np.abs(np.diff(block_mat, axis=1)), axis=1)
    
    # 4. Longest continuous run > 104.0
    longest_runs = np.zeros(n, dtype=np.float32)
    for i in range(n):
        row = block_mat[i]
        mask = row > 104.0
        if not mask.any():
            continue
        # Find longest streak of True
        streak = 0
        max_streak = 0
        for val in mask:
            if val:
                streak += 1
                if streak > max_streak:
                    max_streak = streak
            else:
                streak = 0
        longest_runs[i] = max_streak
        
    return pd.DataFrame({
        'b_mean': b_mean,
        'b_std': b_std,
        'b_min': b_min,
        'b_max': b_max,
        'b_p95': b_p95,
        'b_p99': b_p99,
        'b_cnt_104': cnt_104,
        'b_cnt_106': cnt_106,
        'b_max_spike_dev': max_spike_dev,
        'b_first_diff': first_diff,
        'b_longest_run_104': longest_runs
    })

train_b_stats = compute_block_burst_features(train_blocks)
val_b_stats = compute_block_burst_features(val_blocks)

X_train_B_tab = pd.concat([X_train_A, train_b_stats], axis=1)
X_val_B_tab = pd.concat([X_val_A, val_b_stats], axis=1)
print(f"  • Total Tabular Model B Features: {X_train_B_tab.shape[1]} (518 Param/Spatial + {train_b_stats.shape[1]} Block Signals)", flush=True)

# -----------------------------------------------------------------------------
# 4. TRAIN TABULAR MODEL B STREAM
# -----------------------------------------------------------------------------
print("\n[4/7] Training Tabular Model B Stream (LightGBM on 529 features)...", flush=True)
lgb_tab = lgb.LGBMClassifier(
    n_estimators=650, learning_rate=0.03, num_leaves=45,
    scale_pos_weight=10.0, colsample_bytree=0.75, subsample=0.8,
    random_state=SEED, n_jobs=-1, verbose=-1
)
lgb_tab.fit(X_train_B_tab, y_train)
p_tab_val = lgb_tab.predict_proba(X_val_B_tab)[:, 1]

# -----------------------------------------------------------------------------
# 5. LOAD & EVALUATE 1D CONVNET SEQUENCE STREAM
# -----------------------------------------------------------------------------
print("\n[5/7] Evaluating 1D ConvNet Sequence Stream on 2,000 blocks...", flush=True)
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
cnn_model = Block1DCNN().to(device)

cnn_ckpt_path = os.path.join(PROJECT_ROOT, 'models', 'cnn_1d_block_model.pt')
if os.path.exists(cnn_ckpt_path):
    print(f"  • Loading pre-trained 1D CNN checkpoint from {cnn_ckpt_path} on {device}...", flush=True)
    cnn_model.load_state_dict(torch.load(cnn_ckpt_path, map_location=device))
    cnn_model.eval()
else:
    print("  • Training 1D CNN checkpoint from scratch...", flush=True)
    norm_tr = (train_blocks - 100.0) / 15.0
    norm_val = (val_blocks - 100.0) / 15.0
    # Quick fine-tune
    from src.model_b import train_block_cnn
    cnn_model = train_block_cnn(train_blocks, y_train, val_blocks, y_val, epochs=10, device=device)

# Normalized tensor inputs
norm_val_blocks = (val_blocks - 100.0) / 15.0
tensor_val_x = torch.tensor(norm_val_blocks, dtype=torch.float32).unsqueeze(1)
val_embeds, p_cnn_val = _get_model_outputs_in_batches(cnn_model, tensor_val_x, batch_size=512, device=device)

# -----------------------------------------------------------------------------
# 6. LATE PROBABILITY STACKING (DUAL-STREAM BLEND)
# -----------------------------------------------------------------------------
print("\n[6/7] Building Calibrated Late Probability Stacking...", flush=True)

# Grid search optimal probability blending weight
best_blend_w = 0.70
best_blend_f1 = -1.0
best_blend_p = None

for w in np.arange(0.50, 0.95, 0.05):
    p_blend = w * p_tab_val + (1.0 - w) * p_cnn_val
    p_arr, r_arr, t_arr = precision_recall_curve(y_val, p_blend)
    f1_arr = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(p_arr[:-1], r_arr[:-1])]
    if np.max(f1_arr) > best_blend_f1:
        best_blend_f1 = np.max(f1_arr)
        best_blend_w = w
        best_blend_p = p_blend

print(f"  • Optimal Late Fusion Weight: {best_blend_w:.2f} * Tabular + {1.0-best_blend_w:.2f} * 1D-CNN", flush=True)

# -----------------------------------------------------------------------------
# 7. BENCHMARK COMPARISON TABLE
# -----------------------------------------------------------------------------
def eval_probs(name, probs):
    roc = roc_auc_score(y_val, probs)
    p_arr, r_arr, thresh_arr = precision_recall_curve(y_val, probs)
    pr_auc = auc(r_arr, p_arr)
    
    f1_arr = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(p_arr[:-1], r_arr[:-1])]
    best_idx = np.argmax(f1_arr)
    opt_thresh = thresh_arr[best_idx]
    
    preds = (probs >= opt_thresh).astype(int)
    f1 = f1_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    
    return {
        'name': name,
        'f1': f1,
        'auc_pr': pr_auc,
        'roc_auc': roc,
        'precision': prec,
        'recall': rec,
        'opt_thresh': opt_thresh,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

res_list = [
    eval_probs("Old Model A Baseline (506 Feats)", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, scale_pos_weight=10.0, random_state=SEED, verbose=-1).fit(train_df[feat_cols], y_train).predict_proba(val_df[feat_cols])[:, 1]),
    eval_probs("Tabular Model B (518 Param + Burst Features)", p_tab_val),
    eval_probs("1D CNN Sequence Stream Only (2000 Blocks)", p_cnn_val),
    eval_probs("Advanced Dual-Stream Model B (Late Fusion)", best_blend_p)
]

print("\n" + "=" * 95, flush=True)
print(f"{'Model Architecture':45s} | {'F1':6s} | {'AUC-PR':6s} | {'Recall':7s} | {'Precision':9s} | {'ROC-AUC':7s} | {'TP':4s} | {'FN':4s} | {'FP':4s}", flush=True)
print("=" * 95, flush=True)
for r in res_list:
    print(f"{r['name']:45s} | {r['f1']:.4f} | {r['auc_pr']:.4f} | {r['recall']*100:6.2f}% | {r['precision']*100:8.2f}% | {r['roc_auc']:.4f} | {r['tp']:4d} | {r['fn']:4d} | {r['fp']:4d}", flush=True)
print("=" * 95, flush=True)

print("\n--- ADVANCED DUAL-STREAM MODEL B THRESHOLD & RECALL PROGRESSION ---", flush=True)
print(f"{'Operating Threshold':20s} | {'Recall':7s} | {'Precision':9s} | {'F1':6s} | {'TP (Caught)':12s} | {'FN (Missed)':12s} | {'FP (False Alarms)':18s}", flush=True)
print("-" * 95, flush=True)
for t in [0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]:
    preds_t = (best_blend_p >= t).astype(int)
    rec_t = recall_score(y_val, preds_t, zero_division=0)
    prec_t = precision_score(y_val, preds_t, zero_division=0)
    f1_t = f1_score(y_val, preds_t, zero_division=0)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_val, preds_t).ravel()
    print(f"Threshold = {t:.2f}         | {rec_t*100:6.2f}% | {prec_t*100:8.2f}% | {f1_t:.4f} | {tp_t:12d} | {fn_t:12d} | {fp_t:18d}", flush=True)
print("-" * 95, flush=True)
