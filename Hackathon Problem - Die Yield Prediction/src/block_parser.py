import os
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

def parse_block_matrix(block_series, k=2000):
    """Parse block_readings series into a 2D float32 numpy array [N, k]."""
    n = len(block_series)
    matrix = np.zeros((n, k), dtype=np.float32)
    
    for i, b_str in enumerate(block_series):
        arr = np.fromstring(b_str, sep=' ', dtype=np.float32)
        if len(arr) == k:
            matrix[i] = arr
        else:
            matrix[i, :min(len(arr), k)] = arr[:k]
            
    return matrix

def extract_B0_scalar_features(block_matrix):
    """Extract B0 scalar block summary features (16 features)."""
    b_mean = block_matrix.mean(axis=1)
    b_std = block_matrix.std(axis=1)
    b_min = block_matrix.min(axis=1)
    b_max = block_matrix.max(axis=1)
    
    b_p90 = np.percentile(block_matrix, 90, axis=1)
    b_p95 = np.percentile(block_matrix, 95, axis=1)
    b_p99 = np.percentile(block_matrix, 99, axis=1)
    
    max_minus_100 = b_max - 100.0
    mad_100 = np.mean(np.abs(block_matrix - 100.0), axis=1)
    
    cnt_103 = np.sum(block_matrix > 103.0, axis=1)
    cnt_104 = np.sum(block_matrix > 104.0, axis=1)
    cnt_105 = np.sum(block_matrix > 105.0, axis=1)
    
    # 50 and 100 rolling averages
    roll_50_max = np.zeros(len(block_matrix), dtype=np.float32)
    roll_100_max = np.zeros(len(block_matrix), dtype=np.float32)
    
    for i in range(len(block_matrix)):
        r50 = uniform_filter1d(block_matrix[i], size=50, mode='nearest')
        r100 = uniform_filter1d(block_matrix[i], size=100, mode='nearest')
        roll_50_max[i] = r50.max()
        roll_100_max[i] = r100.max()
        
    cols = [
        'b_mean', 'b_std', 'b_min', 'b_max', 'b_p90', 'b_p95', 'b_p99',
        'max_minus_100', 'mad_100', 'cnt_103', 'cnt_104', 'cnt_105',
        'max_roll50', 'max_roll100', 'max_roll50_dev', 'max_roll100_dev'
    ]
    
    feats = np.column_stack([
        b_mean, b_std, b_min, b_max, b_p90, b_p95, b_p99,
        max_minus_100, mad_100, cnt_103, cnt_104, cnt_105,
        roll_50_max, roll_100_max, roll_50_max - 100.0, roll_100_max - 100.0
    ])
    
    return pd.DataFrame(feats, columns=cols)

def extract_B1_engineered_features(block_matrix):
    """Extract B1 structure-aware cluster and run-length features."""
    b0_df = extract_B0_scalar_features(block_matrix)
    n, k = block_matrix.shape
    
    longest_runs = np.zeros(n, dtype=np.float32)
    num_runs = np.zeros(n, dtype=np.float32)
    cluster_widths = np.zeros(n, dtype=np.float32)
    cluster_areas = np.zeros(n, dtype=np.float32)
    
    thresh = 104.0
    
    for i in range(n):
        row = block_matrix[i]
        anom_mask = (row > thresh)
        
        if np.any(anom_mask):
            # Run length analysis
            diffs = np.diff(np.concatenate(([0], anom_mask.astype(int), [0])))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            run_lens = ends - starts
            
            longest_runs[i] = run_lens.max() if len(run_lens) > 0 else 0
            num_runs[i] = len(run_lens)
            
            # Cluster width and area above threshold
            first_idx = np.where(anom_mask)[0][0]
            last_idx = np.where(anom_mask)[0][-1]
            cluster_widths[i] = last_idx - first_idx + 1
            cluster_areas[i] = np.sum(row[anom_mask] - thresh)
        else:
            longest_runs[i] = 0
            num_runs[i] = 0
            cluster_widths[i] = 0
            cluster_areas[i] = 0
            
    b1_add = pd.DataFrame({
        'longest_run_104': longest_runs,
        'num_runs_104': num_runs,
        'cluster_width': cluster_widths,
        'cluster_area': cluster_areas
    })
    
    return pd.concat([b0_df, b1_add], axis=1)
