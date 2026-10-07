import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter
from sklearn.decomposition import PCA
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

print("=" * 95, flush=True)
print("EVALUATION: ADVANCED SELECTIVE CASCADE ARCHITECTURE (ENHANCED MODEL A + ENHANCED MODEL B)", flush=True)
print("=" * 95, flush=True)

# 1. LOAD DATA
train_path = os.path.join(PROJECT_ROOT, 'input', 'train.csv')
print(f"\n[1/5] Loading full dataset from {train_path}...", flush=True)
raw_df = pd.read_csv(train_path)
eligible_df = raw_df[raw_df['old_label'] == 0].reset_index(drop=True)

train_mask, val_mask, train_wafers, val_wafers = get_wafer_split(eligible_df)
train_df = eligible_df[train_mask].reset_index(drop=True)
val_df = eligible_df[val_mask].reset_index(drop=True)
validate_wafer_partition_integrity(train_df, val_df, train_wafers, val_wafers)

y_train = train_df['label'].values
y_val = val_df['label'].values
feat_cols = [f'feature_{i}' for i in range(1, 501)]

# 2. COMPUTE ADVANCED FEATURES
print("\n[2/5] Computing polar spatial & multivariate PCA features...", flush=True)
def compute_enhanced_spatial(df_subset, full_df):
    spatial_list = []
    for wid, group in df_subset.groupby('wafer_id'):
        rows, cols = group['die_row'].values, group['die_col'].values
        r_max, c_max = max(rows.max(), 1), max(cols.max(), 1)
        cy, cx = r_max / 2.0, c_max / 2.0
        rad = np.sqrt((rows - cy)**2 + (cols - cx)**2)
        theta = np.arctan2(rows - cy, cols - cx)
        
        full_w = full_df[full_df['wafer_id'] == wid]
        grid_r, grid_c = full_w['die_row'].max() + 1, full_w['die_col'].max() + 1
        old_f = np.zeros((grid_r, grid_c), dtype=float)
        val_g = np.zeros((grid_r, grid_c), dtype=float)
        old_f[full_w['die_row'].values, full_w['die_col'].values] = full_w['old_label'].values
        val_g[full_w['die_row'].values, full_w['die_col'].values] = 1.0
        
        def get_d(sz):
            fg = uniform_filter(old_f, size=sz, mode='constant', cval=0.0)
            vg = uniform_filter(val_g, size=sz, mode='constant', cval=0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(vg > 0, fg / vg, 0.0)[rows, cols]
                
        spatial_list.append(pd.DataFrame({
            'radial_dist': rad / (rad.max() if rad.max() > 0 else 1.0),
            'sin_theta': np.sin(theta),
            'cos_theta': np.cos(theta),
            'density_3x3': get_d(3),
            'density_5x5': get_d(5),
            'density_7x7': get_d(7),
            'edge_dist': np.minimum(np.minimum(rows, r_max - rows), np.minimum(cols, c_max - cols)) / (max(r_max, c_max) / 2.0),
            'die_row': rows,
            'die_col': cols,
            'wafer_old_fail_rate': full_w['old_label'].mean()
        }, index=group.index))
    return pd.concat(spatial_list).sort_index()

train_spatial = compute_enhanced_spatial(train_df, raw_df)
val_spatial = compute_enhanced_spatial(val_df, raw_df)

pass_mean = train_df[feat_cols].values[y_train == 0].mean(axis=0)
pass_std = np.maximum(train_df[feat_cols].values[y_train == 0].std(axis=0), 1e-6)
Z_train = (train_df[feat_cols].values - pass_mean) / pass_std
Z_val = (val_df[feat_cols].values - pass_mean) / pass_std

pca = PCA(n_components=25, random_state=SEED).fit(Z_train[y_train == 0])
def extract_pstats(Z_mat):
    abs_z = np.abs(Z_mat)
    Z_proj = pca.transform(Z_mat)
    Z_recon = pca.inverse_transform(Z_proj)
    return pd.DataFrame({
        'param_max_abs_z': np.max(abs_z, axis=1),
        'param_mean_abs_z': np.mean(abs_z, axis=1),
        'param_std_z': np.std(Z_mat, axis=1),
        'param_sum_sq_z': np.sum(Z_mat**2, axis=1),
        'param_cnt_gt_2sigma': np.sum(abs_z > 2.0, axis=1),
        'param_cnt_gt_3sigma': np.sum(abs_z > 3.0, axis=1),
        'pca_recon_error': np.sum((Z_mat - Z_recon)**2, axis=1)
    })

train_pstats = extract_pstats(Z_train)
val_pstats = extract_pstats(Z_val)

X_train_A = pd.concat([train_df[feat_cols], train_spatial, train_pstats], axis=1)
X_val_A = pd.concat([val_df[feat_cols], val_spatial, val_pstats], axis=1)

# 3. PARSE BLOCKS & COMPUTE BURST FEATURES
print("\n[3/5] Parsing 2,000 blocks & computing burst anomaly signals...", flush=True)
def parse_mat(series):
    mat = np.zeros((len(series), 2000), dtype=np.float32)
    for i, s in enumerate(series):
        mat[i] = np.fromstring(s, sep=' ', dtype=np.float32)
    return mat

train_blocks = parse_mat(train_df['block_readings'])
val_blocks = parse_mat(val_df['block_readings'])

def compute_burst(block_mat):
    n = len(block_mat)
    cnt_104 = np.sum(block_mat > 104.0, axis=1)
    cnt_106 = np.sum(block_mat > 106.0, axis=1)
    longest_runs = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mask = block_mat[i] > 104.0
        if not mask.any(): continue
        s, m = 0, 0
        for v in mask:
            if v: s += 1; m = max(m, s)
            else: s = 0
        longest_runs[i] = m
    return pd.DataFrame({
        'b_mean': np.mean(block_mat, axis=1),
        'b_std': np.std(block_mat, axis=1),
        'b_min': np.min(block_mat, axis=1),
        'b_max': np.max(block_mat, axis=1),
        'b_p95': np.percentile(block_mat, 95, axis=1),
        'b_p99': np.percentile(block_mat, 99, axis=1),
        'b_cnt_104': cnt_104,
        'b_cnt_106': cnt_106,
        'b_max_spike_dev': np.max(np.abs(block_mat - 100.0), axis=1),
        'b_first_diff': np.mean(np.abs(np.diff(block_mat, axis=1)), axis=1),
        'b_longest_run_104': longest_runs
    })

train_bstats = compute_burst(train_blocks)
val_bstats = compute_burst(val_blocks)

X_train_B = pd.concat([X_train_A, train_bstats], axis=1)
X_val_B = pd.concat([X_val_A, val_bstats], axis=1)

# 4. TRAIN AND INFER MODELS
print("\n[4/5] Training Enhanced Model A & Model B...", flush=True)
model_A = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=45, scale_pos_weight=10.0, random_state=SEED, verbose=-1)
model_A.fit(X_train_A, y_train)
p_A_val = model_A.predict_proba(X_val_A)[:, 1]

model_B_tab = lgb.LGBMClassifier(n_estimators=650, learning_rate=0.03, num_leaves=45, scale_pos_weight=10.0, random_state=SEED, verbose=-1)
model_B_tab.fit(X_train_B, y_train)
p_B_tab_val = model_B_tab.predict_proba(X_val_B)[:, 1]

# 1D CNN evaluation
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
cnn_model = Block1DCNN().to(device)
cnn_ckpt_path = os.path.join(PROJECT_ROOT, 'models', 'cnn_1d_block_model.pt')
cnn_model.load_state_dict(torch.load(cnn_ckpt_path, map_location=device))
cnn_model.eval()

norm_val_blocks = (val_blocks - 100.0) / 15.0
tensor_val_x = torch.tensor(norm_val_blocks, dtype=torch.float32).unsqueeze(1)
_, p_cnn_val = _get_model_outputs_in_batches(cnn_model, tensor_val_x, batch_size=512, device=device)

# Model B Late Fusion Probability
p_B_val = 0.90 * p_B_tab_val + 0.10 * p_cnn_val

# -----------------------------------------------------------------------------
# 5. SELECTIVE CASCADE ROUTING & GATING EVALUATION
# -----------------------------------------------------------------------------
print("\n[5/5] Evaluating Selective Cascade Gating Strategies...", flush=True)

def eval_cascade(lower_g, upper_g, tau_A=0.51, tau_B=0.52):
    n = len(y_val)
    p_cascade = np.copy(p_A_val)
    preds_cascade = np.zeros(n, dtype=int)
    
    # Cascade decision logic
    escalated_mask = (p_A_val >= lower_g) & (p_A_val <= upper_g)
    
    # Fast path confident passes
    pass_mask = (p_A_val < lower_g)
    preds_cascade[pass_mask] = 0
    p_cascade[pass_mask] = p_A_val[pass_mask]
    
    # Fast path confident fails
    fail_mask = (p_A_val > upper_g)
    preds_cascade[fail_mask] = (p_A_val[fail_mask] >= tau_A).astype(int)
    p_cascade[fail_mask] = p_A_val[fail_mask]
    
    # Escalated uncertain zone -> evaluated by Model B
    p_cascade[escalated_mask] = p_B_val[escalated_mask]
    preds_cascade[escalated_mask] = (p_B_val[escalated_mask] >= tau_B).astype(int)
    
    # Metrics
    f1 = f1_score(y_val, preds_cascade)
    prec = precision_score(y_val, preds_cascade, zero_division=0)
    rec = recall_score(y_val, preds_cascade, zero_division=0)
    roc = roc_auc_score(y_val, p_cascade)
    p_curve, r_curve, _ = precision_recall_curve(y_val, p_cascade)
    auc_pr = auc(r_curve, p_curve)
    tn, fp, fn, tp = confusion_matrix(y_val, preds_cascade).ravel()
    
    esc_count = int(escalated_mask.sum())
    esc_rate = (esc_count / n) * 100
    comp_saved = 100.0 - esc_rate
    
    # Model A baseline misses recovered
    pred_A_base = (p_A_val >= tau_A).astype(int)
    fn_A_mask = (y_val == 1) & (pred_A_base == 0)
    recovered_fns = int((preds_cascade[fn_A_mask] == 1).sum())
    
    return {
        'window': f"[{lower_g:.2f}, {upper_g:.2f}]",
        'f1': f1, 'auc_pr': auc_pr, 'roc_auc': roc,
        'prec': prec, 'rec': rec,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'esc_count': esc_count,
        'esc_rate': esc_rate,
        'comp_saved': comp_saved,
        'recovered_fns': recovered_fns
    }

cascade_results = [
    eval_cascade(0.25, 0.55, tau_A=0.51, tau_B=0.52),
    eval_cascade(0.20, 0.50, tau_A=0.51, tau_B=0.50),
    eval_cascade(0.18, 0.52, tau_A=0.51, tau_B=0.48),
    eval_cascade(0.30, 0.60, tau_A=0.51, tau_B=0.52),
]

print("\n" + "=" * 115, flush=True)
print(f"{'Selective Cascade Window':26s} | {'F1':6s} | {'AUC-PR':6s} | {'Recall':7s} | {'Precision':9s} | {'ROC-AUC':7s} | {'TP':4s} | {'FN':4s} | {'FP':4s} | {'Escalated':9s} | {'Compute Saved':13s}")
print("=" * 115, flush=True)
for c in cascade_results:
    print(f"{c['window']:26s} | {c['f1']:.4f} | {c['auc_pr']:.4f} | {c['rec']*100:6.2f}% | {c['prec']*100:8.2f}% | {c['roc_auc']:.4f} | {c['tp']:4d} | {c['fn']:4d} | {c['fp']:4d} | {c['esc_count']:4d} ({c['esc_rate']:4.1f}%) | {c['comp_saved']:6.2f}%", flush=True)
print("=" * 115, flush=True)
