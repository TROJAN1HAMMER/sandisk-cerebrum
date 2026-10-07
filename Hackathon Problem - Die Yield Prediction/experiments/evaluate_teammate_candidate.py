#!/usr/bin/env python3
"""
Comprehensive Head-to-Head Benchmark: Teammate Candidate vs B2 Baseline
Evaluates on:
1. Canonical Holdout (32 wafers, N=27,972, 1,056 defects)
2. Independent Test Set EXP-04 (40 wafers, N=32,598, 1,380 defects)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, confusion_matrix, brier_score_loss,
    matthews_corrcoef, fbeta_score
)
import lightgbm as lgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_wafer_split
from src.evaluation_protocol import validate_wafer_partition_integrity
from src.model_b import Block1DCNN, _get_model_outputs_in_batches
from src.spatial import compute_spatial_features
from src.block_parser import parse_block_matrix, extract_B1_engineered_features

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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
                
        edge_dist = np.minimum(np.minimum(rows, r_max - rows), np.minimum(cols, c_max - cols)) / (max(r_max, c_max) / 2.0)
        spatial_list.append(pd.DataFrame({
            'radial_dist': rad / (rad.max() if rad.max() > 0 else 1.0),
            'sin_theta': np.sin(theta),
            'cos_theta': np.cos(theta),
            'density_3x3': get_d(3),
            'density_5x5': get_d(5),
            'density_7x7': get_d(7),
            'edge_dist': edge_dist,
            'die_row': rows,
            'die_col': cols,
            'wafer_old_fail_rate': full_w['old_label'].mean()
        }, index=group.index))
    return pd.concat(spatial_list).sort_index()

def compute_block_burst(block_mat):
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

def extract_advanced_candidate_block_features(block_mat):
    n = len(block_mat)
    b_mean = block_mat.mean(axis=1)
    b_max = block_mat.max(axis=1)
    b_p95 = np.percentile(block_mat, 95, axis=1)
    b_cnt_104 = (block_mat > 104.0).sum(axis=1)
    b_cnt_106 = (block_mat > 106.0).sum(axis=1)
    b_dev_100 = np.abs(block_mat - 100.0).max(axis=1)
    b_diff = np.abs(np.diff(block_mat, axis=1)).mean(axis=1)
    
    longest_runs = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mask = block_mat[i] > 104.0
        if not mask.any():
            continue
        cur = 0
        mx = 0
        for v in mask:
            if v:
                cur += 1
                if cur > mx: mx = cur
            else:
                cur = 0
        longest_runs[i] = mx

    w10 = uniform_filter(block_mat, size=(1, 10), mode='nearest')
    w50 = uniform_filter(block_mat, size=(1, 50), mode='nearest')
    b_roll10_p95 = np.percentile(w10, 95, axis=1)
    b_roll50_p95 = np.percentile(w50, 95, axis=1)

    return pd.DataFrame({
        'b_mean': b_mean,
        'b_max': b_max,
        'b_p95': b_p95,
        'b_cnt_gt_104': b_cnt_104,
        'b_cnt_gt_106': b_cnt_106,
        'b_dev_from_100': b_dev_100,
        'b_first_diff_mean': b_diff,
        'b_longest_run_gt_104': longest_runs,
        'b_roll10_p95': b_roll10_p95,
        'b_roll50_p95': b_roll50_p95,
    })

def full_metrics(y_true, probs, preds):
    f1 = float(f1_score(y_true, preds, zero_division=0))
    f2 = float(fbeta_score(y_true, preds, beta=2, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, preds))
    roc = float(roc_auc_score(y_true, probs))
    p_arr, r_arr, _ = precision_recall_curve(y_true, probs)
    pr_auc = float(auc(r_arr, p_arr))
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    brier = float(brier_score_loss(y_true, probs))
    ece = float(calculate_ece(y_true, probs))
    return {
        'f1': round(f1, 4),
        'f2': round(f2, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'mcc': round(mcc, 4),
        'pr_auc': round(pr_auc, 4),
        'roc_auc': round(roc, 4),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'brier': round(brier, 4),
        'ece': round(ece, 4)
    }

def main():
    print("=" * 80)
    print("EVALUATING TEAMMATE CANDIDATE VS B2 BASELINE ON CANONICAL & EXP-04")
    print("=" * 80)

    # 1. Load Train & Val
    train_path = os.path.join(PROJECT_ROOT, 'input', 'train.csv')
    print(f"\n[1] Loading training and canonical validation data...")
    raw_df = pd.read_csv(train_path)
    eligible_df = raw_df[raw_df['old_label'] == 0].reset_index(drop=True)
    train_mask, val_mask, train_wafers, val_wafers = get_wafer_split(eligible_df)
    train_df = eligible_df[train_mask].reset_index(drop=True)
    val_df = eligible_df[val_mask].reset_index(drop=True)

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    feat_cols = [f'feature_{i}' for i in range(1, 501)]

    # 2. Enhanced features for Train & Val
    print(f"\n[2] Computing teammate features (spatial + PCA moments)...")
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

    X_train_A_enh = pd.concat([train_df[feat_cols], train_spatial, train_pstats], axis=1)
    X_val_A_enh = pd.concat([val_df[feat_cols], val_spatial, val_pstats], axis=1)

    # Blocks from cache
    print(f"\n[3] Loading block readings from exp_cache...")
    train_blocks = np.load(os.path.join(PROJECT_ROOT, 'experiments', 'exp_cache', 'tr_blocks_2000.npy'))
    val_blocks = np.load(os.path.join(PROJECT_ROOT, 'experiments', 'exp_cache', 'val_blocks_2000.npy'))

    train_bstats = compute_block_burst(train_blocks)
    val_bstats = compute_block_burst(val_blocks)

    X_train_B_enh = pd.concat([X_train_A_enh, train_bstats], axis=1)
    X_val_B_enh = pd.concat([X_val_A_enh, val_bstats], axis=1)

    # 3. Train Candidate Models or Load if already cached
    candidates_dir = os.path.join(PROJECT_ROOT, 'models', 'candidates')
    os.makedirs(candidates_dir, exist_ok=True)
    model_A_path = os.path.join(candidates_dir, 'model_A_enhanced.pkl')
    model_B_path = os.path.join(candidates_dir, 'model_B_tab_enhanced.pkl')

    if os.path.exists(model_A_path) and os.path.exists(model_B_path):
        print(f"\n[4] Loading cached candidate models from {candidates_dir}...")
        with open(model_A_path, 'rb') as f:
            cand_A = pickle.load(f)
        with open(model_B_path, 'rb') as f:
            cand_B = pickle.load(f)
    else:
        print(f"\n[4] Training Candidate Enhanced Models...")
        cand_A = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=45, scale_pos_weight=10.0, random_state=SEED, verbose=-1, n_jobs=1)
        cand_A.fit(X_train_A_enh, y_train)
        with open(model_A_path, 'wb') as f:
            pickle.dump(cand_A, f)

        cand_B = lgb.LGBMClassifier(n_estimators=650, learning_rate=0.03, num_leaves=45, scale_pos_weight=10.0, random_state=SEED, verbose=-1, n_jobs=1)
        cand_B.fit(X_train_B_enh, y_train)
        with open(model_B_path, 'wb') as f:
            pickle.dump(cand_B, f)

    # CNN inference for val
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    cnn_model = Block1DCNN().to(device)
    cnn_ckpt_path = os.path.join(PROJECT_ROOT, 'models', 'cnn_1d_block_model.pt')
    cnn_model.load_state_dict(torch.load(cnn_ckpt_path, map_location=device))
    cnn_model.eval()

    norm_val_blocks = (val_blocks - 100.0) / 15.0
    tensor_val_x = torch.tensor(norm_val_blocks, dtype=torch.float32).unsqueeze(1)
    _, p_cnn_val = _get_model_outputs_in_batches(cnn_model, tensor_val_x, batch_size=512, device=device)

    p_val_A_enh = cand_A.predict_proba(X_val_A_enh)[:, 1]
    p_val_B_tab_enh = cand_B.predict_proba(X_val_B_enh)[:, 1]
    p_val_B_enh = 0.90 * p_val_B_tab_enh + 0.10 * p_cnn_val

    # Evaluate Candidate on Canonical Holdout
    print("\n--- CANONICAL HOLDOUT CANDIDATE REPRODUCTION ---")
    # Standalone A
    pred_val_A_cand = (p_val_A_enh >= 0.51).astype(int)
    m_val_A_cand = full_metrics(y_val, p_val_A_enh, pred_val_A_cand)
    print(f"Candidate Model A (tau=0.51): F1={m_val_A_cand['f1']:.4f}, Rec={m_val_A_cand['recall']*100:.2f}%, Prec={m_val_A_cand['precision']*100:.2f}%, PR-AUC={m_val_A_cand['pr_auc']:.4f}, TP={m_val_A_cand['tp']}, FP={m_val_A_cand['fp']}")

    # Standalone B
    p_arr, r_arr, t_arr = precision_recall_curve(y_val, p_val_B_enh)
    f1_arr = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(p_arr[:-1], r_arr[:-1])]
    opt_t_B = float(t_arr[np.argmax(f1_arr)])
    pred_val_B_cand = (p_val_B_enh >= opt_t_B).astype(int)
    m_val_B_cand = full_metrics(y_val, p_val_B_enh, pred_val_B_cand)
    print(f"Candidate Model B (opt tau={opt_t_B:.4f}): F1={m_val_B_cand['f1']:.4f}, Rec={m_val_B_cand['recall']*100:.2f}%, Prec={m_val_B_cand['precision']*100:.2f}%, PR-AUC={m_val_B_cand['pr_auc']:.4f}, TP={m_val_B_cand['tp']}, FP={m_val_B_cand['fp']}")

    # Enhanced Cascade: Gate [0.25, 0.55], tau_A=0.51, tau_B=0.52
    esc_mask_val = (p_val_A_enh >= 0.25) & (p_val_A_enh <= 0.55)
    p_val_casc_cand = np.where(esc_mask_val, p_val_B_enh, p_val_A_enh)
    pred_val_casc_cand = np.zeros(len(y_val), dtype=int)
    pred_val_casc_cand[p_val_A_enh < 0.25] = 0
    pred_val_casc_cand[p_val_A_enh > 0.55] = (p_val_A_enh[p_val_A_enh > 0.55] >= 0.51).astype(int)
    pred_val_casc_cand[esc_mask_val] = (p_val_B_enh[esc_mask_val] >= 0.52).astype(int)
    m_val_casc_cand = full_metrics(y_val, p_val_casc_cand, pred_val_casc_cand)
    esc_rate_val = esc_mask_val.mean() * 100.0
    print(f"Enhanced Cascade ([0.25, 0.55], tau_A=0.51, tau_B=0.52): F1={m_val_casc_cand['f1']:.4f}, Rec={m_val_casc_cand['recall']*100:.2f}%, Prec={m_val_casc_cand['precision']*100:.2f}%, PR-AUC={m_val_casc_cand['pr_auc']:.4f}, TP={m_val_casc_cand['tp']}, FP={m_val_casc_cand['fp']}, Compute Saved={100.0 - esc_rate_val:.2f}%")

    # High-Recall Cascade: Gate [0.20, 0.50], tau_A=0.51, tau_B=0.50
    esc_mask_val_hr = (p_val_A_enh >= 0.20) & (p_val_A_enh <= 0.50)
    p_val_casc_hr = np.where(esc_mask_val_hr, p_val_B_enh, p_val_A_enh)
    pred_val_casc_hr = np.zeros(len(y_val), dtype=int)
    pred_val_casc_hr[p_val_A_enh < 0.20] = 0
    pred_val_casc_hr[p_val_A_enh > 0.50] = (p_val_A_enh[p_val_A_enh > 0.50] >= 0.51).astype(int)
    pred_val_casc_hr[esc_mask_val_hr] = (p_val_B_enh[esc_mask_val_hr] >= 0.50).astype(int)
    m_val_casc_hr = full_metrics(y_val, p_val_casc_hr, pred_val_casc_hr)
    esc_rate_val_hr = esc_mask_val_hr.mean() * 100.0
    print(f"High Recall Cascade ([0.20, 0.50], tau_A=0.51, tau_B=0.50): F1={m_val_casc_hr['f1']:.4f}, Rec={m_val_casc_hr['recall']*100:.2f}%, Prec={m_val_casc_hr['precision']*100:.2f}%, PR-AUC={m_val_casc_hr['pr_auc']:.4f}, TP={m_val_casc_hr['tp']}, FP={m_val_casc_hr['fp']}, Compute Saved={100.0 - esc_rate_val_hr:.2f}%")

    # 4. LOAD INDEPENDENT TEST SET EXP-04 (input/test.csv)
    print("\n" + "=" * 80)
    print("[5] EVALUATING ON INDEPENDENT EXP-04 DATASET (input/test.csv)...")
    print("=" * 80)
    test_csv_path = os.path.join(PROJECT_ROOT, 'input', 'test.csv')
    df_raw_test = pd.read_csv(test_csv_path)
    eligible_test_mask = (df_raw_test['old_label'] == 0)
    eligible_test_df = df_raw_test[eligible_test_mask].reset_index(drop=True)
    y_test = eligible_test_df['label'].values
    n_test = len(y_test)
    print(f"Loaded {n_test:,} eligible dies across {eligible_test_df['wafer_id'].nunique()} unseen wafers. True defects: {y_test.sum():,}")

    # Extract teammate features for test set
    print("Extracting teammate features for test set...")
    test_spatial_enh = compute_enhanced_spatial(eligible_test_df, df_raw_test)
    Z_test = (eligible_test_df[feat_cols].values - pass_mean) / pass_std
    test_pstats_enh = extract_pstats(Z_test)

    X_test_A_enh = pd.concat([eligible_test_df[feat_cols], test_spatial_enh, test_pstats_enh], axis=1)

    # Blocks for test
    print("Parsing test blocks...")
    test_blocks = parse_block_matrix(eligible_test_df['block_readings'])
    test_bstats_enh = compute_block_burst(test_blocks)
    X_test_B_enh = pd.concat([X_test_A_enh, test_bstats_enh], axis=1)

    # 1D CNN inference on test
    norm_test_blocks = (test_blocks - 100.0) / 15.0
    tensor_test_x = torch.tensor(norm_test_blocks, dtype=torch.float32).unsqueeze(1)
    _, p_cnn_test = _get_model_outputs_in_batches(cnn_model, tensor_test_x, batch_size=512, device=device)

    # Candidate inference on test
    t0 = time.perf_counter()
    p_test_A_enh = cand_A.predict_proba(X_test_A_enh)[:, 1]
    t_test_A_cand = (time.perf_counter() - t0) / n_test * 1000.0

    t0 = time.perf_counter()
    p_test_B_tab_enh = cand_B.predict_proba(X_test_B_enh)[:, 1]
    p_test_B_enh = 0.90 * p_test_B_tab_enh + 0.10 * p_cnn_test
    t_test_B_cand = (time.perf_counter() - t0) / n_test * 1000.0

    # 5. LOAD PRODUCTION B2 MODEL & EVALUATE ON TEST SET
    print("\n[6] Evaluating Frozen Production B2 on same test set...")
    with open(os.path.join(PROJECT_ROOT, 'models', 'model_A.pkl'), 'rb') as f:
        prod_A = pickle.load(f)
    with open(os.path.join(PROJECT_ROOT, 'models', 'model_B.pkl'), 'rb') as f:
        prod_B = pickle.load(f)

    sp_test_prod = compute_spatial_features(eligible_test_df, full_df=df_raw_test)
    spatial_cols_prod = ['radial_dist', 'density_3x3', 'density_5x5', 'edge_dist', 'die_row', 'die_col']
    X_test_A_prod = pd.concat([eligible_test_df[feat_cols], sp_test_prod[spatial_cols_prod]], axis=1)

    b1_test_prod = extract_B1_engineered_features(test_blocks)
    cand_block_test_prod = extract_advanced_candidate_block_features(test_blocks)
    sp2_prod = sp_test_prod[['radial_dist', 'edge_dist']].rename(
        columns={'radial_dist': 'radial_distance_norm', 'edge_dist': 'edge_distance_norm'}
    )
    X_test_B_prod = pd.concat([eligible_test_df[feat_cols], b1_test_prod, cand_block_test_prod, sp2_prod], axis=1)

    t0 = time.perf_counter()
    p_test_A_prod = prod_A.predict_proba(X_test_A_prod)[:, 1]
    t_test_A_prod = (time.perf_counter() - t0) / n_test * 1000.0

    t0 = time.perf_counter()
    p_test_B_prod = prod_B.predict_proba(X_test_B_prod)[:, 1]
    t_test_B_prod = (time.perf_counter() - t0) / n_test * 1000.0

    # Evaluate Prod B2 Cascade on Test
    esc_mask_prod = (p_test_A_prod >= 0.295) & (p_test_A_prod <= 0.550)
    p_test_casc_prod = np.where(esc_mask_prod, p_test_B_prod, p_test_A_prod)
    pred_test_casc_prod = np.where(
        esc_mask_prod,
        (p_test_B_prod >= 0.5376).astype(int),
        (p_test_A_prod >= 0.5100).astype(int)
    )
    m_test_casc_prod = full_metrics(y_test, p_test_casc_prod, pred_test_casc_prod)
    esc_rate_test_prod = float(esc_mask_prod.mean() * 100.0)
    compute_saved_test_prod = 100.0 - esc_rate_test_prod

    # Evaluate Candidate Cascade on Test (Teammate config: Gate [0.25, 0.55], tau_A=0.51, tau_B=0.52)
    esc_mask_cand = (p_test_A_enh >= 0.25) & (p_test_A_enh <= 0.55)
    p_test_casc_cand = np.where(esc_mask_cand, p_test_B_enh, p_test_A_enh)
    pred_test_casc_cand = np.zeros(n_test, dtype=int)
    pred_test_casc_cand[p_test_A_enh < 0.25] = 0
    pred_test_casc_cand[p_test_A_enh > 0.55] = (p_test_A_enh[p_test_A_enh > 0.55] >= 0.51).astype(int)
    pred_test_casc_cand[esc_mask_cand] = (p_test_B_enh[esc_mask_cand] >= 0.52).astype(int)
    m_test_casc_cand = full_metrics(y_test, p_test_casc_cand, pred_test_casc_cand)
    esc_rate_test_cand = float(esc_mask_cand.mean() * 100.0)
    compute_saved_test_cand = 100.0 - esc_rate_test_cand

    # Evaluate Candidate High-Recall Cascade on Test (Teammate config: Gate [0.20, 0.50], tau_A=0.51, tau_B=0.50)
    esc_mask_cand_hr = (p_test_A_enh >= 0.20) & (p_test_A_enh <= 0.50)
    p_test_casc_cand_hr = np.where(esc_mask_cand_hr, p_test_B_enh, p_test_A_enh)
    pred_test_casc_cand_hr = np.zeros(n_test, dtype=int)
    pred_test_casc_cand_hr[p_test_A_enh < 0.20] = 0
    pred_test_casc_cand_hr[p_test_A_enh > 0.50] = (p_test_A_enh[p_test_A_enh > 0.50] >= 0.51).astype(int)
    pred_test_casc_cand_hr[esc_mask_cand_hr] = (p_test_B_enh[esc_mask_cand_hr] >= 0.50).astype(int)
    m_test_casc_cand_hr = full_metrics(y_test, p_test_casc_cand_hr, pred_test_casc_cand_hr)
    esc_rate_test_cand_hr = float(esc_mask_cand_hr.mean() * 100.0)
    compute_saved_test_cand_hr = 100.0 - esc_rate_test_cand_hr

    # Evaluate Standalone Model A Prod vs Candidate on Test
    m_test_A_prod = full_metrics(y_test, p_test_A_prod, (p_test_A_prod >= 0.5100).astype(int))
    m_test_A_cand = full_metrics(y_test, p_test_A_enh, (p_test_A_enh >= 0.5100).astype(int))

    # Evaluate Standalone Model B Prod vs Candidate on Test
    m_test_B_prod = full_metrics(y_test, p_test_B_prod, (p_test_B_prod >= 0.5376).astype(int))
    m_test_B_cand = full_metrics(y_test, p_test_B_enh, (p_test_B_enh >= 0.5200).astype(int))

    # Rescue analysis for B2
    fn_A_prod = (y_test == 1) & (p_test_A_prod < 0.5100)
    esc_and_fn_A_prod = esc_mask_prod & fn_A_prod
    fn_rescued_prod = int((pred_test_casc_prod[esc_and_fn_A_prod] == 1).sum())
    fn_sent_prod = int(esc_and_fn_A_prod.sum())

    tp_A_prod = (y_test == 1) & (p_test_A_prod >= 0.5100)
    esc_and_tp_A_prod = esc_mask_prod & tp_A_prod
    tp_lost_prod = int((pred_test_casc_prod[esc_and_tp_A_prod] == 0).sum())
    net_tp_prod = fn_rescued_prod - tp_lost_prod

    # Rescue analysis for Candidate
    fn_A_cand = (y_test == 1) & (p_test_A_enh < 0.5100)
    esc_and_fn_A_cand = esc_mask_cand & fn_A_cand
    fn_rescued_cand = int((pred_test_casc_cand[esc_and_fn_A_cand] == 1).sum())
    fn_sent_cand = int(esc_and_fn_A_cand.sum())

    tp_A_cand = (y_test == 1) & (p_test_A_enh >= 0.5100)
    esc_and_tp_A_cand = esc_mask_cand & tp_A_cand
    tp_lost_cand = int((pred_test_casc_cand[esc_and_tp_A_cand] == 0).sum())
    net_tp_cand = fn_rescued_cand - tp_lost_cand

    # Summary dictionary
    summary = {
        'canonical_holdout': {
            'b2_cascade': {
                'f1': 0.5254, 'precision': 0.8932, 'recall': 0.3722, 'pr_auc': 0.4777,
                'roc_auc': 0.7986, 'tp': 393, 'fp': 47, 'fn': 663, 'tn': 26869, 'compute_saved': 94.02
            },
            'candidate_enhanced_cascade': {
                **m_val_casc_cand, 'compute_saved': round(100.0 - esc_rate_val, 2), 'esc_count': int(esc_mask_val.sum())
            },
            'candidate_high_recall_cascade': {
                **m_val_casc_hr, 'compute_saved': round(100.0 - esc_rate_val_hr, 2), 'esc_count': int(esc_mask_val_hr.sum())
            },
            'candidate_model_A': m_val_A_cand,
            'candidate_model_B': m_val_B_cand
        },
        'independent_exp04': {
            'b2_cascade': {
                **m_test_casc_prod, 'compute_saved': round(compute_saved_test_prod, 2),
                'esc_count': int(esc_mask_prod.sum()), 'esc_rate': round(esc_rate_test_prod, 2),
                'fn_rescued': fn_rescued_prod, 'tp_lost': tp_lost_prod, 'net_tp': net_tp_prod,
                'latency_A_ms': round(t_test_A_prod, 4), 'latency_B_ms': round(t_test_B_prod, 4)
            },
            'candidate_enhanced_cascade': {
                **m_test_casc_cand, 'compute_saved': round(compute_saved_test_cand, 2),
                'esc_count': int(esc_mask_cand.sum()), 'esc_rate': round(esc_rate_test_cand, 2),
                'fn_rescued': fn_rescued_cand, 'tp_lost': tp_lost_cand, 'net_tp': net_tp_cand,
                'latency_A_ms': round(t_test_A_cand, 4), 'latency_B_ms': round(t_test_B_cand, 4)
            },
            'candidate_high_recall_cascade': {
                **m_test_casc_hr, 'compute_saved': round(compute_saved_test_cand_hr, 2),
                'esc_count': int(esc_mask_cand_hr.sum()), 'esc_rate': round(esc_rate_test_cand_hr, 2)
            },
            'model_A_comparison': {
                'prod_b2_A': m_test_A_prod,
                'cand_enh_A': m_test_A_cand
            },
            'model_B_comparison': {
                'prod_b2_B': m_test_B_prod,
                'cand_enh_B': m_test_B_cand
            }
        }
    }

    out_file = os.path.join(PROJECT_ROOT, 'results', 'teammate_candidate_evaluation.json')
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 90)
    print("FINAL HEAD-TO-HEAD SUMMARY ON INDEPENDENT EXP-04 (input/test.csv)")
    print("=" * 90)
    print(f"{'Metric':<20} | {'Current B2':<15} | {'Candidate Enhanced':<20} | {'Candidate High-Rec':<20}")
    print("-" * 90)
    for k in ['f1', 'f2', 'precision', 'recall', 'mcc', 'pr_auc', 'roc_auc', 'tp', 'fp', 'fn', 'compute_saved', 'brier', 'ece']:
        v_b2 = summary['independent_exp04']['b2_cascade'].get(k)
        v_c1 = summary['independent_exp04']['candidate_enhanced_cascade'].get(k)
        v_c2 = summary['independent_exp04']['candidate_high_recall_cascade'].get(k)
        print(f"{k:<20} | {str(v_b2):<15} | {str(v_c1):<20} | {str(v_c2):<20}")
    print("=" * 90)

if __name__ == '__main__':
    main()
