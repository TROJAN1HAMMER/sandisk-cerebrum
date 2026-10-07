#!/usr/bin/env python3
"""
Fast, Rigorous Independent Validation of Teammate Candidate Models on EXP-04 (input/test.csv)
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

from src.model_b import Block1DCNN, _get_model_outputs_in_batches
from src.block_parser import parse_block_matrix

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

def compute_block_burst_fast(block_mat):
    n = len(block_mat)
    cnt_104 = np.sum(block_mat > 104.0, axis=1)
    cnt_106 = np.sum(block_mat > 106.0, axis=1)
    longest_runs = np.zeros(n, dtype=np.float32)
    has_spikes = np.where(cnt_104 > 0)[0]
    for i in has_spikes:
        m = (block_mat[i] > 104.0).view(np.int8)
        p = np.diff(np.concatenate(([0], m, [0])))
        starts = np.where(p == 1)[0]
        ends = np.where(p == -1)[0]
        longest_runs[i] = np.max(ends - starts)
        
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
    print("INDEPENDENT EXP-04 EVALUATION: TEAMMATE CANDIDATE VS FROZEN B2")
    print("=" * 80)

    # 1. Fit PCA & Normalization on Training Passing Dies from Cache
    print("\n[1/5] Loading training parametric pass distribution from exp_cache...")
    X_tr_A = np.load(os.path.join(PROJECT_ROOT, 'experiments', 'exp_cache', 'X_tr_A.npy'))
    y_tr = np.load(os.path.join(PROJECT_ROOT, 'experiments', 'exp_cache', 'y_tr.npy'))
    raw_feats_tr = X_tr_A[:, :500]
    pass_mask_tr = (y_tr == 0)
    pass_mean = raw_feats_tr[pass_mask_tr].mean(axis=0)
    pass_std = np.maximum(raw_feats_tr[pass_mask_tr].std(axis=0), 1e-6)

    Z_tr_pass = (raw_feats_tr[pass_mask_tr] - pass_mean) / pass_std
    pca = PCA(n_components=25, random_state=SEED).fit(Z_tr_pass)

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

    # Save PCA pipeline into models/candidates
    with open(os.path.join(PROJECT_ROOT, 'models', 'candidates', 'pca_pipeline.pkl'), 'wb') as f:
        pickle.dump({'pca': pca, 'pass_mean': pass_mean, 'pass_std': pass_std}, f)

    # 2. Load Candidate Models
    print("\n[2/5] Loading Candidate Enhanced Models from models/candidates/...")
    with open(os.path.join(PROJECT_ROOT, 'models', 'candidates', 'model_A_enhanced.pkl'), 'rb') as f:
        cand_A = pickle.load(f)
    with open(os.path.join(PROJECT_ROOT, 'models', 'candidates', 'model_B_tab_enhanced.pkl'), 'rb') as f:
        cand_B = pickle.load(f)

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    cnn_model = Block1DCNN().to(device)
    cnn_ckpt_path = os.path.join(PROJECT_ROOT, 'models', 'cnn_1d_block_model.pt')
    cnn_model.load_state_dict(torch.load(cnn_ckpt_path, map_location=device))
    cnn_model.eval()

    # 3. Load input/test.csv
    print("\n[3/5] Loading input/test.csv (EXP-04 Independent Test Set)...")
    test_path = os.path.join(PROJECT_ROOT, 'input', 'test.csv')
    df_raw_test = pd.read_csv(test_path)
    eligible_test_mask = (df_raw_test['old_label'] == 0)
    eligible_test_df = df_raw_test[eligible_test_mask].reset_index(drop=True)
    y_test = eligible_test_df['label'].values
    n_test = len(y_test)
    feat_cols = [f'feature_{i}' for i in range(1, 501)]
    print(f"  • Total eligible dies: {n_test:,} across {eligible_test_df['wafer_id'].nunique()} unseen wafers. Defect dies: {y_test.sum():,}")

    # 4. Extract Features
    print("\n[4/5] Extracting teammate candidate features on test set...")
    t0 = time.time()
    test_spatial = compute_enhanced_spatial(eligible_test_df, df_raw_test)
    Z_test = (eligible_test_df[feat_cols].values - pass_mean) / pass_std
    test_pstats = extract_pstats(Z_test)
    X_test_A_cand = pd.concat([eligible_test_df[feat_cols], test_spatial, test_pstats], axis=1)

    print("  • Parsing block sequences...")
    test_blocks = parse_block_matrix(eligible_test_df['block_readings'])
    test_bstats = compute_block_burst_fast(test_blocks)
    X_test_B_cand = pd.concat([X_test_A_cand, test_bstats], axis=1)
    print(f"  • Feature extraction complete in {time.time() - t0:.2f}s.")

    # 5. Model Inference
    print("\n[5/5] Running model inference on EXP-04...")
    t0 = time.perf_counter()
    p_test_A_cand = cand_A.predict_proba(X_test_A_cand)[:, 1]
    lat_A = (time.perf_counter() - t0) / n_test * 1000.0

    t0 = time.perf_counter()
    p_test_B_tab_cand = cand_B.predict_proba(X_test_B_cand)[:, 1]
    norm_test_blocks = (test_blocks - 100.0) / 15.0
    tensor_test_x = torch.tensor(norm_test_blocks, dtype=torch.float32).unsqueeze(1)
    _, p_cnn_test = _get_model_outputs_in_batches(cnn_model, tensor_test_x, batch_size=512, device=device)
    p_test_B_cand = 0.90 * p_test_B_tab_cand + 0.10 * p_cnn_test
    lat_B = (time.perf_counter() - t0) / n_test * 1000.0

    # Standalone metrics on EXP-04
    pred_test_A_cand = (p_test_A_cand >= 0.5100).astype(int)
    m_test_A_cand = full_metrics(y_test, p_test_A_cand, pred_test_A_cand)

    pred_test_B_cand = (p_test_B_cand >= 0.5200).astype(int)
    m_test_B_cand = full_metrics(y_test, p_test_B_cand, pred_test_B_cand)

    # Cascade 1: Enhanced Cascade [0.25, 0.55], tau_A=0.51, tau_B=0.52
    esc_mask_c1 = (p_test_A_cand >= 0.25) & (p_test_A_cand <= 0.55)
    p_test_casc_c1 = np.where(esc_mask_c1, p_test_B_cand, p_test_A_cand)
    pred_test_casc_c1 = np.zeros(n_test, dtype=int)
    pred_test_casc_c1[p_test_A_cand < 0.25] = 0
    pred_test_casc_c1[p_test_A_cand > 0.55] = (p_test_A_cand[p_test_A_cand > 0.55] >= 0.51).astype(int)
    pred_test_casc_c1[esc_mask_c1] = (p_test_B_cand[esc_mask_c1] >= 0.52).astype(int)
    m_test_casc_c1 = full_metrics(y_test, p_test_casc_c1, pred_test_casc_c1)
    esc_rate_c1 = float(esc_mask_c1.mean() * 100.0)

    # Cascade 2: High Recall Cascade [0.20, 0.50], tau_A=0.51, tau_B=0.50
    esc_mask_c2 = (p_test_A_cand >= 0.20) & (p_test_A_cand <= 0.50)
    p_test_casc_c2 = np.where(esc_mask_c2, p_test_B_cand, p_test_A_cand)
    pred_test_casc_c2 = np.zeros(n_test, dtype=int)
    pred_test_casc_c2[p_test_A_cand < 0.20] = 0
    pred_test_casc_c2[p_test_A_cand > 0.50] = (p_test_A_cand[p_test_A_cand > 0.50] >= 0.51).astype(int)
    pred_test_casc_c2[esc_mask_c2] = (p_test_B_cand[esc_mask_c2] >= 0.50).astype(int)
    m_test_casc_c2 = full_metrics(y_test, p_test_casc_c2, pred_test_casc_c2)
    esc_rate_c2 = float(esc_mask_c2.mean() * 100.0)

    # Load frozen B2 metrics
    with open(os.path.join(PROJECT_ROOT, 'experiments', 'EXP-04', 'independent_metrics.json'), 'r') as f:
        b2_meta = json.load(f)

    # Build final comparative summary
    comparison = {
        'evaluation_population': 'Independent Ground-Truth Test Set (input/test.csv)',
        'wafers_count': 40,
        'total_dies': n_test,
        'actual_defects': int(y_test.sum()),
        'b2_production_cascade': {
            **b2_meta['cascade'],
            'escalation_rate_pct': b2_meta['escalation_rate_pct'],
            'compute_saved_pct': b2_meta['compute_saved_pct']
        },
        'candidate_enhanced_cascade': {
            **m_test_casc_c1,
            'escalation_rate_pct': round(esc_rate_c1, 4),
            'compute_saved_pct': round(100.0 - esc_rate_c1, 4),
            'escalated_dies': int(esc_mask_c1.sum()),
            'gate': '[0.25, 0.55]',
            'tau_A': 0.51,
            'tau_B': 0.52,
            'latency_A_ms': round(lat_A, 4),
            'latency_B_ms': round(lat_B, 4)
        },
        'candidate_high_recall_cascade': {
            **m_test_casc_c2,
            'escalation_rate_pct': round(esc_rate_c2, 4),
            'compute_saved_pct': round(100.0 - esc_rate_c2, 4),
            'escalated_dies': int(esc_mask_c2.sum()),
            'gate': '[0.20, 0.50]',
            'tau_A': 0.51,
            'tau_B': 0.50
        },
        'standalone_model_A': {
            'b2_model_A': b2_meta['model_A_baseline'],
            'candidate_model_A': m_test_A_cand
        },
        'standalone_model_B': {
            'b2_model_B': b2_meta['model_B_standalone'],
            'candidate_model_B': m_test_B_cand
        }
    }

    results_path = os.path.join(PROJECT_ROOT, 'results', 'teammate_candidate_exp04_results.json')
    with open(results_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 105)
    print("AUTHORITATIVE INDEPENDENT EXP-04 HEAD-TO-HEAD COMPARISON (40 UNSEEN WAFERS, 32,598 DIES, 1,380 DEFECTS)")
    print("=" * 105)
    print(f"{'Metric':<22} | {'Current B2 Cascade':<20} | {'Candidate Enhanced':<22} | {'Candidate High-Rec':<20} | {'Delta (Cand vs B2)':<15}")
    print("-" * 105)

    metrics_keys = [
        ('f1', 'F1-Score'),
        ('f2', 'F2-Score'),
        ('mcc', 'MCC'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('pr_auc', 'PR-AUC'),
        ('roc_auc', 'ROC-AUC'),
        ('tp', 'True Positives (TP)'),
        ('fp', 'False Positives (FP)'),
        ('fn', 'False Negatives (FN)'),
        ('compute_saved_pct', 'Compute Saved %'),
        ('escalation_rate_pct', 'Escalation Rate %'),
        ('brier', 'Brier Score'),
        ('ece', 'ECE Calibration')
    ]

    for k, label in metrics_keys:
        v_b2 = comparison['b2_production_cascade'].get(k)
        v_c1 = comparison['candidate_enhanced_cascade'].get(k)
        v_c2 = comparison['candidate_high_recall_cascade'].get(k)
        if isinstance(v_b2, (int, float)) and isinstance(v_c1, (int, float)):
            delta = v_c1 - v_b2
            delta_str = f"{delta:+.4f}" if isinstance(delta, float) else f"{delta:+d}"
            v_b2_str = f"{v_b2:.4f}" if isinstance(v_b2, float) else f"{v_b2}"
            v_c1_str = f"{v_c1:.4f}" if isinstance(v_c1, float) else f"{v_c1}"
            v_c2_str = f"{v_c2:.4f}" if isinstance(v_c2, float) else f"{v_c2}"
        else:
            delta_str = "-"
            v_b2_str = str(v_b2)
            v_c1_str = str(v_c1)
            v_c2_str = str(v_c2)
        print(f"{label:<22} | {v_b2_str:<20} | {v_c1_str:<22} | {v_c2_str:<20} | {delta_str:<15}")
    print("=" * 105)

if __name__ == '__main__':
    main()
