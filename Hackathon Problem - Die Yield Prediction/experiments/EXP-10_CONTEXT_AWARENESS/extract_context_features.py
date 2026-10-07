#!/usr/bin/env python3
"""
EXP-10: Context-Aware Feature Extraction Module.
Implements Families A through E with strict zero-leakage compliance.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import convolve

FAM_A_COLS = [
    'valid_neighbors_3x3',
    'f171_local_mean_3x3',
    'f171_local_diff_3x3',
    'f171_local_zscore_3x3',
    'f460_local_diff_3x3',
    'f336_local_diff_3x3',
    'local_electrical_contrast'
]

FAM_B_COLS = [
    'h_defect_density_5x5',
    'v_defect_density_5x5',
    'diag_defect_density_5x5',
    'cluster_aspect_ratio',
    'isolated_defect_flag',
    'dense_cluster_flag'
]

FAM_C_COLS = [
    'f171_wafer_zscore',
    'f171_wafer_robust_z',
    'f460_wafer_zscore',
    'f336_wafer_zscore',
    'f33_wafer_zscore',
    'f109_wafer_zscore',
    'wafer_mean_f171',
    'wafer_std_f171'
]

FAM_D_COLS = [
    'anomaly_peak_pos',
    'anomaly_centroid',
    'anomaly_early_ratio',
    'anomaly_mid_ratio',
    'anomaly_late_ratio',
    'anomaly_asymmetry',
    'sharp_peak_ratio',
    'sustained_anomaly_flag'
]

FAM_E_COLS = [
    'elec_spatial_agreement',
    'elec_signal_agreement',
    'spatial_signal_agreement',
    'concordant_evidence_count',
    'full_concordance_flag'
]

def extract_family_A(w_ids, rows, cols, f171, f460, f336):
    """
    Family A: Local physical neighbourhood (3x3 grid) for top electrical drivers.
    Returns array of shape (N, 7).
    """
    n = len(w_ids)
    out = np.zeros((n, 7), dtype=np.float32)
    k3 = np.ones((3, 3), dtype=np.float32)
    
    unique_wafers = np.unique(w_ids)
    for wid in unique_wafers:
        mask = (w_ids == wid)
        w_rows = rows[mask]
        w_cols = cols[mask]
        w_f171 = f171[mask]
        w_f460 = f460[mask]
        w_f336 = f336[mask]
        
        R = max(int(w_rows.max()) + 1, 85)
        C = max(int(w_cols.max()) + 1, 85)
        
        grid_valid = np.zeros((R, C), dtype=np.float32)
        grid_f171 = np.zeros((R, C), dtype=np.float32)
        grid_f460 = np.zeros((R, C), dtype=np.float32)
        grid_f336 = np.zeros((R, C), dtype=np.float32)
        
        grid_valid[w_rows, w_cols] = 1.0
        grid_f171[w_rows, w_cols] = w_f171
        grid_f460[w_rows, w_cols] = w_f460
        grid_f336[w_rows, w_cols] = w_f336
        
        v_cnt = convolve(grid_valid, k3, mode='constant', cval=0.0)
        
        # Neighbours strictly excluding the die itself
        cnt_nbrs = np.maximum(0.0, v_cnt - grid_valid)
        
        sum_f171 = convolve(grid_f171, k3, mode='constant', cval=0.0) - grid_f171
        sum_sq_f171 = convolve(grid_f171**2, k3, mode='constant', cval=0.0) - grid_f171**2
        
        m_f171 = np.where(cnt_nbrs > 0, sum_f171 / np.maximum(cnt_nbrs, 1e-5), grid_f171)
        var_f171 = np.maximum(0.0, np.where(cnt_nbrs > 0, sum_sq_f171 / np.maximum(cnt_nbrs, 1e-5), grid_f171**2) - m_f171**2)
        std_f171 = np.sqrt(var_f171)
        
        sum_f460 = convolve(grid_f460, k3, mode='constant', cval=0.0) - grid_f460
        m_f460 = np.where(cnt_nbrs > 0, sum_f460 / np.maximum(cnt_nbrs, 1e-5), grid_f460)
        
        sum_f336 = convolve(grid_f336, k3, mode='constant', cval=0.0) - grid_f336
        m_f336 = np.where(cnt_nbrs > 0, sum_f336 / np.maximum(cnt_nbrs, 1e-5), grid_f336)
        
        # Sample at die locations
        nbr_cnt_d = cnt_nbrs[w_rows, w_cols]
        m_171_d = m_f171[w_rows, w_cols]
        diff_171_d = w_f171 - m_171_d
        z_171_d = diff_171_d / (std_f171[w_rows, w_cols] + 1e-5)
        
        diff_460_d = w_f460 - m_f460[w_rows, w_cols]
        diff_336_d = w_f336 - m_f336[w_rows, w_cols]
        
        contrast_d = np.sqrt(diff_171_d**2 + diff_460_d**2 + diff_336_d**2)
        
        out[mask, 0] = nbr_cnt_d
        out[mask, 1] = m_171_d
        out[mask, 2] = diff_171_d
        out[mask, 3] = z_171_d
        out[mask, 4] = diff_460_d
        out[mask, 5] = diff_336_d
        out[mask, 6] = contrast_d
        
    return out

def extract_family_B(w_ids, rows, cols, full_wafer_df, d3, d5):
    """
    Family B: Spatial topology and morphology (5x5 directional filters on old_label screening map).
    Returns array of shape (N, 6).
    """
    n = len(w_ids)
    out = np.zeros((n, 6), dtype=np.float32)
    
    k_h = np.ones((1, 5), dtype=np.float32)
    k_v = np.ones((5, 1), dtype=np.float32)
    k_diag = (np.eye(5, dtype=np.float32) + np.fliplr(np.eye(5, dtype=np.float32)))
    k_diag[2, 2] = 1.0
    
    unique_wafers = np.unique(w_ids)
    for wid in unique_wafers:
        mask = (w_ids == wid)
        w_rows = rows[mask]
        w_cols = cols[mask]
        
        fw = full_wafer_df[full_wafer_df['wafer_id'] == wid]
        fw_rows = fw['die_row'].values
        fw_cols = fw['die_col'].values
        fw_old = fw['old_label'].values.astype(np.float32)
        
        R = max(int(fw_rows.max()) + 1, 85)
        C = max(int(fw_cols.max()) + 1, 85)
        
        grid_fail = np.zeros((R, C), dtype=np.float32)
        grid_valid = np.zeros((R, C), dtype=np.float32)
        grid_fail[fw_rows, fw_cols] = fw_old
        grid_valid[fw_rows, fw_cols] = 1.0
        
        # Horizontal
        h_f = convolve(grid_fail, k_h, mode='constant', cval=0.0)
        h_v = convolve(grid_valid, k_h, mode='constant', cval=0.0)
        h_dens = np.where(h_v > 0, h_f / np.maximum(h_v, 1e-5), 0.0)
        
        # Vertical
        v_f = convolve(grid_fail, k_v, mode='constant', cval=0.0)
        v_v = convolve(grid_valid, k_v, mode='constant', cval=0.0)
        v_dens = np.where(v_v > 0, v_f / np.maximum(v_v, 1e-5), 0.0)
        
        # Diagonal
        d_f = convolve(grid_fail, k_diag, mode='constant', cval=0.0)
        d_v = convolve(grid_valid, k_diag, mode='constant', cval=0.0)
        d_dens = np.where(d_v > 0, d_f / np.maximum(d_v, 1e-5), 0.0)
        
        h_d = h_dens[w_rows, w_cols]
        v_d = v_dens[w_rows, w_cols]
        diag_d = d_dens[w_rows, w_cols]
        
        aspect_ratio = np.log(h_d + 0.01) - np.log(v_d + 0.01)
        
        out[mask, 0] = h_d
        out[mask, 1] = v_d
        out[mask, 2] = diag_d
        out[mask, 3] = aspect_ratio
        
    # Flags using existing d3 and d5
    out[:, 4] = ((d3 == 0.0) & (d5 > 0.0)).astype(np.float32)
    out[:, 5] = (d5 >= 0.40).astype(np.float32)
    
    return out

def extract_family_C(w_ids, f171, f460, f336, f33, f109):
    """
    Family C: Wafer-relative electrical context (mean, std, z-scores, robust z-score).
    Returns array of shape (N, 8).
    """
    n = len(w_ids)
    out = np.zeros((n, 8), dtype=np.float32)
    
    unique_wafers = np.unique(w_ids)
    for wid in unique_wafers:
        mask = (w_ids == wid)
        w_f171 = f171[mask]
        w_f460 = f460[mask]
        w_f336 = f336[mask]
        w_f33 = f33[mask]
        w_f109 = f109[mask]
        
        # f171 stats
        m171 = float(np.mean(w_f171))
        s171 = float(np.std(w_f171))
        med171 = float(np.median(w_f171))
        mad171 = float(np.median(np.abs(w_f171 - med171)))
        
        z171 = (w_f171 - m171) / (s171 + 1e-5)
        rob_z171 = (w_f171 - med171) / (1.4826 * mad171 + 1e-5)
        
        # other drivers z-scores
        z460 = (w_f460 - np.mean(w_f460)) / (np.std(w_f460) + 1e-5)
        z336 = (w_f336 - np.mean(w_f336)) / (np.std(w_f336) + 1e-5)
        z33 = (w_f33 - np.mean(w_f33)) / (np.std(w_f33) + 1e-5)
        z109 = (w_f109 - np.mean(w_f109)) / (np.std(w_f109) + 1e-5)
        
        out[mask, 0] = z171
        out[mask, 1] = rob_z171
        out[mask, 2] = z460
        out[mask, 3] = z336
        out[mask, 4] = z33
        out[mask, 5] = z109
        out[mask, 6] = m171
        out[mask, 7] = s171
        
    return out

def extract_family_D(block_matrix, longest_run_104):
    """
    Family D: Signal geometry and anomaly location along 2,000 blocks.
    Returns array of shape (N, 8).
    """
    n = len(block_matrix)
    out = np.zeros((n, 8), dtype=np.float32)
    
    anom = np.maximum(0.0, block_matrix - 100.0)
    tot_anom = anom.sum(axis=1)
    
    # 1. Peak position
    peak_idx = np.argmax(block_matrix, axis=1)
    out[:, 0] = (peak_idx / 2000.0).astype(np.float32)
    
    # 2. Centroid
    idx_arr = np.arange(2000, dtype=np.float32)
    centroid = (anom @ idx_arr) / (tot_anom + 1e-5) / 2000.0
    out[:, 1] = centroid.astype(np.float32)
    
    # 3. Early / Mid / Late area ratios
    out[:, 2] = (anom[:, :667].sum(axis=1) / (tot_anom + 1e-5)).astype(np.float32)
    out[:, 3] = (anom[:, 667:1334].sum(axis=1) / (tot_anom + 1e-5)).astype(np.float32)
    out[:, 4] = (anom[:, 1334:].sum(axis=1) / (tot_anom + 1e-5)).astype(np.float32)
    
    # 4. Asymmetry (late vs early)
    out[:, 5] = ((anom[:, 1000:].sum(axis=1) - anom[:, :1000].sum(axis=1)) / (tot_anom + 1e-5)).astype(np.float32)
    
    # 5. Sharp peak ratio (window +/- 10 around peak)
    offsets = np.arange(-10, 11)
    idx_matrix = np.clip(peak_idx[:, None] + offsets[None, :], 0, 1999)
    row_indices = np.arange(n)[:, None]
    peak_area = anom[row_indices, idx_matrix].sum(axis=1)
    out[:, 6] = (peak_area / (tot_anom + 1e-5)).astype(np.float32)
    
    # 6. Sustained anomaly flag
    out[:, 7] = (longest_run_104 >= 50).astype(np.float32)
    
    return out

def extract_family_E(f171_wafer_zscore, density_5x5, cluster_area_110):
    """
    Family E: Evidence agreement and cross-sensor concordance.
    Returns array of shape (N, 5).
    """
    n = len(f171_wafer_zscore)
    out = np.zeros((n, 5), dtype=np.float32)
    
    elec_anom = np.maximum(0.0, f171_wafer_zscore)
    spatial_anom = density_5x5
    signal_anom = np.log1p(np.maximum(0.0, cluster_area_110))
    
    # Pairwise interactions
    out[:, 0] = elec_anom * spatial_anom
    out[:, 1] = elec_anom * signal_anom
    out[:, 2] = spatial_anom * signal_anom
    
    # Count of active sensors
    c_elec = (f171_wafer_zscore > 1.5).astype(np.float32)
    c_spat = (density_5x5 > 0.10).astype(np.float32)
    c_sig = (cluster_area_110 > 0.0).astype(np.float32)
    
    conc_count = c_elec + c_spat + c_sig
    out[:, 3] = conc_count
    out[:, 4] = (conc_count == 3.0).astype(np.float32)
    
    return out
